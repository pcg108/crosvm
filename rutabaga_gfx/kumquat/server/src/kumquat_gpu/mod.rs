// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap as Map;
use std::collections::BTreeSet as Set;
use std::fs::File;
use std::io::Cursor;
use std::io::Write;
use std::os::fd::AsFd;
use std::os::fd::BorrowedFd;
use std::os::raw::c_void;
use std::sync::Arc;
use std::sync::Mutex;
use std::ptr;
use nix::sys::mman::{ProtFlags, MapFlags, mmap};
use nix::fcntl::{open, OFlag};
use std::ffi::CString;
use nix::sys::stat::Mode;
use nix::unistd::ftruncate;
use std::num::NonZeroUsize;
use std::os::fd::OwnedFd;
use std::os::fd::FromRawFd;
use std::sync::OnceLock;
use nix::libc;
use once_cell::sync::Lazy;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::io::Seek;
use std::io::Read;
use std::path::Path;
use std::io::SeekFrom;

use log::error;
use rutabaga_gfx::calculate_capset_mask;
use rutabaga_gfx::kumquat_support::kumquat_gpu_protocol::*;
use rutabaga_gfx::kumquat_support::RutabagaEvent;
use rutabaga_gfx::kumquat_support::RutabagaMemoryMapping;
use rutabaga_gfx::kumquat_support::RutabagaSharedMemory;
use rutabaga_gfx::kumquat_support::RutabagaStream;
use rutabaga_gfx::kumquat_support::RutabagaTube;
use rutabaga_gfx::ResourceCreate3D;
use rutabaga_gfx::ResourceCreateBlob;
use rutabaga_gfx::Rutabaga;
use rutabaga_gfx::RutabagaBuilder;
use rutabaga_gfx::RutabagaComponentType;
use rutabaga_gfx::RutabagaDescriptor;
use rutabaga_gfx::RutabagaError;
use rutabaga_gfx::RutabagaFence;
use rutabaga_gfx::RutabagaFenceHandler;
use rutabaga_gfx::RutabagaFromRawDescriptor;
use rutabaga_gfx::RutabagaHandle;
use rutabaga_gfx::RutabagaIntoRawDescriptor;
use rutabaga_gfx::RutabagaIovec;
use rutabaga_gfx::RutabagaResult;
use rutabaga_gfx::RutabagaWsi;
use rutabaga_gfx::Transfer3D;
use rutabaga_gfx::VulkanInfo;
use rutabaga_gfx::RUTABAGA_FLAG_FENCE;
use rutabaga_gfx::RUTABAGA_FLAG_FENCE_HOST_SHAREABLE;
use rutabaga_gfx::RUTABAGA_MAP_ACCESS_RW;
use rutabaga_gfx::RUTABAGA_MAP_CACHE_CACHED;
use rutabaga_gfx::RUTABAGA_MEM_HANDLE_TYPE_SHM;
use rutabaga_gfx::RutabagaMapping;
use rutabaga_gfx::RutabagaGralloc;
use rutabaga_gfx::RutabagaGrallocBackendFlags;
use rutabaga_gfx::RutabagaMappedRegion;

const VK_ICD_FILENAMES: &str = "VK_ICD_FILENAMES";

const SNAPSHOT_DIR: &str = "/tmp/";

pub struct XdmaDevice {
    xdma_read: File,
    xdma_write: File,
}

impl XdmaDevice {
    /// Create a new XDMA device instance
    pub fn new() -> Self {
        let read_path = format!("/dev/xdma0_c2h_0");
        let write_path = format!("/dev/xdma0_h2c_0");

        let read_file = OpenOptions::new()
            .read(true)
            .open(read_path)
            .expect("Error opening xdma read");

        let write_file = OpenOptions::new()
            .write(true)
            .open(write_path)
            .expect("Error opening xdma write");

        XdmaDevice {
            xdma_read: read_file,
            xdma_write: write_file,
        }
    }

    pub fn read_data(&mut self, buffer: &mut [u8], offset: u64) -> Result<usize, std::io::Error> {
        self.xdma_read.seek(std::io::SeekFrom::Start(offset))?;
        self.xdma_read.read(buffer)
    }

    pub fn read_bytes(&mut self, num_bytes: usize, offset: u64) -> Result<Vec<u8>, std::io::Error> {
        let mut buffer = vec![0u8; num_bytes];
        self.read_data(&mut buffer, offset)?;
        Ok(buffer)
    }

    pub fn write_bytes(&mut self, offset: u64, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
        self.xdma_write.seek(SeekFrom::Start(offset))?;
        self.xdma_write.write(buffer)
    }
}


pub struct KumquatGpuConnection {
    stream: RutabagaStream,
    connection_id: u64,
    xdma_device: XdmaDevice,
}

pub struct KumquatGpuResource {
    attached_contexts: Set<u32>,
    mapping: Option<RutabagaMemoryMapping>,
    opt_mapping: Option<Box<dyn RutabagaMappedRegion>>,
}

pub struct FenceData {
    pub pending_fences: Map<u64, RutabagaEvent>,
}

pub type FenceState = Arc<Mutex<FenceData>>;

pub fn create_fence_handler(fence_state: FenceState) -> RutabagaFenceHandler {
    RutabagaFenceHandler::new(move |completed_fence: RutabagaFence| {
        let mut state = fence_state.lock().unwrap();
        match (*state).pending_fences.entry(completed_fence.fence_id) {
            Entry::Occupied(o) => {
                let (_, mut event) = o.remove_entry();
                event.signal().unwrap();
            }
            Entry::Vacant(_) => {
                // This is fine, since an actual fence doesn't create emulated sync
                // entry
            }
        }
    })
}

fn gralloc() -> &'static Mutex<RutabagaGralloc> {
    static GRALLOC: OnceLock<Mutex<RutabagaGralloc>> = OnceLock::new();
    GRALLOC.get_or_init(|| {
        // The idea is to make sure the gfxstream ICD isn't loaded when gralloc starts
        // up. The Nvidia ICD should be loaded.
        //
        // This is mostly useful for developers.  For AOSP hermetic gfxstream end2end
        // testing, VK_ICD_FILENAMES shouldn't be defined.  For deqp-vk, this is
        // useful, but not safe for multi-threaded tests.  For now, since this is only
        // used for end2end tests, we should be good.
        let vk_icd_name_opt = match std::env::var(VK_ICD_FILENAMES) {
            Ok(vk_icd_name) => {
                std::env::remove_var(VK_ICD_FILENAMES);
                Some(vk_icd_name)
            }
            Err(_) => None,
        };

        let gralloc = Mutex::new(RutabagaGralloc::new(RutabagaGrallocBackendFlags::new()).unwrap());

        if let Some(vk_icd_name) = vk_icd_name_opt {
            std::env::set_var(VK_ICD_FILENAMES, vk_icd_name);
        }

        gralloc
    })
}

pub struct KumquatGpu {
    rutabaga: Rutabaga,
    fence_state: FenceState,
    snapshot_buffer: Cursor<Vec<u8>>,
    id_allocator: u32,
    resources: Map<u32, KumquatGpuResource>,
}


const DMA_ADDR: u64 = (0x8800_0000 + 0x3_8000_0000) % 0x4_0000_0000;
const DMA_SIZE: usize = 100_000_000; 


impl KumquatGpu {
    pub fn new(capset_names: String, renderer_features: String) -> RutabagaResult<KumquatGpu> {
        println!("New Kumquat GPU");

        let capset_mask = calculate_capset_mask(capset_names.as_str().split(":"));
        let fence_state = Arc::new(Mutex::new(FenceData {
            pending_fences: Default::default(),
        }));

        let fence_handler = create_fence_handler(fence_state.clone());

        let renderer_features_opt = if renderer_features.is_empty() {
            None
        } else {
            Some(renderer_features)
        };

        let rutabaga = RutabagaBuilder::new(RutabagaComponentType::CrossDomain, capset_mask)
            .set_use_external_blob(true)
            .set_use_egl(true)
            .set_wsi(RutabagaWsi::Surfaceless)
            .set_renderer_features(renderer_features_opt)
            .build(fence_handler, None)?;

        Ok(KumquatGpu {
            rutabaga,
            fence_state,
            snapshot_buffer: Cursor::new(Vec::new()),
            id_allocator: 0,
            resources: Default::default(),
        })
    }

    pub fn allocate_id(&mut self) -> u32 {
        self.id_allocator = self.id_allocator + 1;
        self.id_allocator
    }
}

impl KumquatGpuConnection {
    pub fn new(connection: RutabagaTube, connection_id: u64) -> KumquatGpuConnection {

        let mut xdma_device = XdmaDevice::new();

        KumquatGpuConnection {
            stream: RutabagaStream::new(connection),
            connection_id,
            xdma_device,
        }
    }

    pub fn process_command(&mut self, kumquat_gpu: &mut KumquatGpu) -> RutabagaResult<bool> {
        let mut hung_up = false;
        let protocols = self.stream.read()?;

        for protocol in protocols {
            match protocol {
                KumquatGpuProtocol::GetNumCapsets => {
                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_NUM_CAPSETS,
                        payload: kumquat_gpu.rutabaga.get_num_capsets(),
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::GetCapsetInfo(capset_index) => {
                    let (capset_id, version, size) =
                        kumquat_gpu.rutabaga.get_capset_info(capset_index)?;

                    let resp = kumquat_gpu_protocol_resp_capset_info {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_CAPSET_INFO,
                            ..Default::default()
                        },
                        capset_id,
                        version,
                        size,
                        ..Default::default()
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::GetCapset(cmd) => {
                    let capset = kumquat_gpu
                        .rutabaga
                        .get_capset(cmd.capset_id, cmd.capset_version)?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_CAPSET,
                        payload: capset.len().try_into()?,
                    };

                    self.stream
                        .write(KumquatGpuProtocolWrite::CmdWithData(resp, capset))?;
                }
                KumquatGpuProtocol::GetConnectionId => {

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_CONNECTION_ID,
                        payload: self.connection_id as u32,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;

                }
                KumquatGpuProtocol::CtxCreate(cmd) => {
                    let context_id = kumquat_gpu.allocate_id();
                    let context_name: Option<String> =
                        String::from_utf8(cmd.debug_name.to_vec()).ok();

                    kumquat_gpu.rutabaga.create_context(
                        context_id,
                        cmd.context_init,
                        context_name.as_deref(),
                    )?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_CONTEXT_CREATE,
                        payload: context_id,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::CtxDestroy(ctx_id) => {
                    kumquat_gpu.rutabaga.destroy_context(ctx_id)?;
                }
                KumquatGpuProtocol::CtxAttachResource(cmd) => {
                    kumquat_gpu
                        .rutabaga
                        .context_attach_resource(cmd.ctx_id, cmd.resource_id)?;
                }
                KumquatGpuProtocol::CtxDetachResource(cmd) => {
                    kumquat_gpu
                        .rutabaga
                        .context_detach_resource(cmd.ctx_id, cmd.resource_id)?;

                    let mut resource = kumquat_gpu
                        .resources
                        .remove(&cmd.resource_id)
                        .ok_or(RutabagaError::InvalidResourceId)?;

                    resource.attached_contexts.remove(&cmd.ctx_id);
                    if resource.attached_contexts.len() == 0 {
                        if resource.mapping.is_some() {
                            kumquat_gpu.rutabaga.detach_backing(cmd.resource_id)?;
                        }

                        kumquat_gpu.rutabaga.unref_resource(cmd.resource_id)?;
                    } else {
                        kumquat_gpu.resources.insert(cmd.resource_id, resource);
                    }
                }
                KumquatGpuProtocol::ResourceCreate3d(cmd) => {
                    let resource_create_3d = ResourceCreate3D {
                        target: cmd.target,
                        format: cmd.format,
                        bind: cmd.bind,
                        width: cmd.width,
                        height: cmd.height,
                        depth: cmd.depth,
                        array_size: cmd.array_size,
                        last_level: cmd.last_level,
                        nr_samples: cmd.nr_samples,
                        flags: cmd.flags,
                    };

                    let size = cmd.size as usize;
                    let descriptor: RutabagaDescriptor =
                        RutabagaSharedMemory::new("rutabaga_server", size as u64)?.into();

                    let clone = descriptor.try_clone()?;
                    let mut vecs: Vec<RutabagaIovec> = Vec::new();

                    // Creating the mapping closes the cloned descriptor.
                    let mapping = RutabagaMemoryMapping::from_safe_descriptor(
                        clone,
                        size,
                        RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
                    )?;
                    let rutabaga_mapping = mapping.as_rutabaga_mapping();

                    vecs.push(RutabagaIovec {
                        base: rutabaga_mapping.ptr as *mut c_void,
                        len: size,
                    });

                    let resource_id = kumquat_gpu.allocate_id();

                    kumquat_gpu
                        .rutabaga
                        .resource_create_3d(resource_id, resource_create_3d)?;

                    kumquat_gpu.rutabaga.attach_backing(resource_id, vecs)?;
                    kumquat_gpu.resources.insert(
                        resource_id,
                        KumquatGpuResource {
                            attached_contexts: Default::default(),
                            mapping: Some(mapping),
                            opt_mapping: None,
                        },
                    );

                    kumquat_gpu
                        .rutabaga
                        .context_attach_resource(cmd.ctx_id, resource_id)?;

                    let resp = kumquat_gpu_protocol_resp_resource_create {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_RESOURCE_CREATE,
                            ..Default::default()
                        },
                        resource_id,
                        ..Default::default()
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;

                    // self.stream.write(KumquatGpuProtocolWrite::CmdWithHandle(
                    //     resp,
                    //     RutabagaHandle {
                    //         os_handle: descriptor,
                    //         handle_type: RUTABAGA_MEM_HANDLE_TYPE_SHM,
                    //     },
                    // ))?;
                }
                KumquatGpuProtocol::TransferToHost3d(cmd) => {
                    let resource_id = cmd.resource_id;

                    let transfer = Transfer3D {
                        x: cmd.box_.x,
                        y: cmd.box_.y,
                        z: cmd.box_.z,
                        w: cmd.box_.w,
                        h: cmd.box_.h,
                        d: cmd.box_.d,
                        level: cmd.level,
                        stride: cmd.stride,
                        layer_stride: cmd.layer_stride,
                        offset: cmd.offset,
                    };

                    kumquat_gpu
                        .rutabaga
                        .transfer_write(cmd.ctx_id, resource_id, transfer)?;

                    // // SAFETY: Safe because the emulated fence and owned by us.
                    // let mut file = unsafe {
                    //     File::from_raw_descriptor(emulated_fence.os_handle.into_raw_descriptor())
                    // };

                    // // TODO(b/356504311): An improvement would be `impl From<RutabagaHandle> for
                    // // RutabagaEvent` + `RutabagaEvent::signal`
                    // file.write(&mut 1u64.to_ne_bytes())?;

                    let resp = kumquat_gpu_protocol_resp_transfer_to_host_3d {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_TRANSFER_TO_HOST_3D,
                            ..Default::default()
                        },
                    };
                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::TransferFromHost3d(cmd) => {
                    let resource_id = cmd.resource_id;

                    let transfer = Transfer3D {
                        x: cmd.box_.x,
                        y: cmd.box_.y,
                        z: cmd.box_.z,
                        w: cmd.box_.w,
                        h: cmd.box_.h,
                        d: cmd.box_.d,
                        level: cmd.level,
                        stride: cmd.stride,
                        layer_stride: cmd.layer_stride,
                        offset: cmd.offset,
                    };

                    kumquat_gpu
                        .rutabaga
                        .transfer_read(cmd.ctx_id, resource_id, transfer, None)?;

                    // // SAFETY: Safe because the emulated fence and owned by us.
                    // let mut file = unsafe {
                    //     File::from_raw_descriptor(emulated_fence.os_handle.into_raw_descriptor())
                    // };

                    // // TODO(b/356504311): An improvement would be `impl From<RutabagaHandle> for
                    // // RutabagaEvent` + `RutabagaEvent::signal`
                    // file.write(&mut 1u64.to_ne_bytes())?;

                    let resp = kumquat_gpu_protocol_resp_transfer_from_host_3d {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_TRANSFER_FROM_HOST_3D,
                            ..Default::default()
                        },
                    };
                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::CmdSubmit3d(cmd, mut cmd_buf, fence_ids) => {
                    kumquat_gpu.rutabaga.submit_command(
                        cmd.ctx_id,
                        &mut cmd_buf[..],
                        &fence_ids[..],
                    )?;

                    if cmd.flags & RUTABAGA_FLAG_FENCE != 0 {
                        let fence_id = kumquat_gpu.allocate_id() as u64;
                        let fence = RutabagaFence {
                            flags: cmd.flags,
                            fence_id,
                            ctx_id: cmd.ctx_id,
                            ring_idx: cmd.ring_idx,
                        };

                        let mut fence_descriptor_opt: Option<RutabagaHandle> = None;
                        let actual_fence = cmd.flags & RUTABAGA_FLAG_FENCE_HOST_SHAREABLE != 0;
                        if !actual_fence {
                            let event: RutabagaEvent = RutabagaEvent::new()?;
                            let clone = event.try_clone()?;
                            let emulated_fence: RutabagaHandle = clone.into();

                            fence_descriptor_opt = Some(emulated_fence);
                            let mut fence_state = kumquat_gpu.fence_state.lock().unwrap();
                            (*fence_state).pending_fences.insert(fence_id, event);
                        }

                        kumquat_gpu.rutabaga.create_fence(fence)?;

                        if actual_fence {
                            fence_descriptor_opt =
                                Some(kumquat_gpu.rutabaga.export_fence(fence_id)?);
                            kumquat_gpu.rutabaga.destroy_fences(&[fence_id])?;
                        }

                        let fence_descriptor = fence_descriptor_opt
                            .ok_or(RutabagaError::SpecViolation("No fence descriptor"))?;

                        let resp = kumquat_gpu_protocol_resp_cmd_submit_3d {
                            hdr: kumquat_gpu_protocol_ctrl_hdr {
                                type_: KUMQUAT_GPU_PROTOCOL_RESP_CMD_SUBMIT_3D,
                                ..Default::default()
                            },
                            fence_id,
                            handle_type: fence_descriptor.handle_type,
                            ..Default::default()
                        };

                        self.stream.write(KumquatGpuProtocolWrite::CmdWithHandle(
                            resp,
                            fence_descriptor,
                        ))?;
                    }
                }
                KumquatGpuProtocol::ResourceCreateBlob(cmd) => {
                    let resource_id = kumquat_gpu.allocate_id();

                    let resource_create_blob = ResourceCreateBlob {
                        blob_mem: cmd.blob_mem,
                        blob_flags: cmd.blob_flags,
                        blob_id: cmd.blob_id,
                        size: cmd.size,
                    };

                    kumquat_gpu.rutabaga.resource_create_blob(
                        cmd.ctx_id,
                        resource_id,
                        resource_create_blob,
                        None,
                        None,
                    )?;

                    let handle = kumquat_gpu.rutabaga.export_blob(resource_id)?;
                    println!("creating resource: {:?}, host handle: {:?}", resource_id, handle);

                    let mut vk_info: VulkanInfo = Default::default();
                    if let Ok(vulkan_info) = kumquat_gpu.rutabaga.vulkan_info(resource_id) {
                        vk_info = vulkan_info;
                    }
                    
                    let clone = handle.try_clone()?;
                    let region = gralloc().lock().unwrap().import_and_map(
                        clone,
                        vk_info,
                        cmd.size as u64,
                    )?;

                    // let clone = handle.try_clone()?;
                    // let resource_memory_mapping = RutabagaMemoryMapping::from_safe_descriptor(
                    //     clone.os_handle,
                    //     cmd.size as usize,
                    //     RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
                    // )?;
                    
                    kumquat_gpu.resources.insert(
                        resource_id,
                        KumquatGpuResource {
                            attached_contexts: Set::from([cmd.ctx_id]),
                            mapping: None, // Some(resource_memory_mapping)
                            opt_mapping: Some(region),
                        },
                    );

                    let resp = kumquat_gpu_protocol_resp_resource_create {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_RESOURCE_CREATE,
                            ..Default::default()
                        },
                        resource_id,
                        handle_type: handle.handle_type,
                        vulkan_info: vk_info,
                    };

                    self.stream
                        .write(KumquatGpuProtocolWrite::CmdWithHandle(resp, handle))?;

                    kumquat_gpu
                        .rutabaga
                        .context_attach_resource(cmd.ctx_id, resource_id)?;

                    println!("created resource: {:?}", resource_id);
                }
                KumquatGpuProtocol::CopyIntoCopyBuffer(cmd) => {
                    // host->guest: copy resource to XDMA

                    // get the rutabaga resource corresponding to this resource-id
                    let resource_id = cmd.resource_id as u32;
                    let resource_size = cmd.resource_size;
                    let is_gpu_resource = cmd.is_gpu_resource;

                    let resource = kumquat_gpu
                        .resources
                        .get(&resource_id)
                        .ok_or(RutabagaError::InvalidResourceId)?;

                    let res_rutabaga_mapping = if is_gpu_resource == 1 {
                        resource.opt_mapping
                            .as_ref()
                            .expect("Error retrieving GPU memory mapping for host->guest")
                            .as_rutabaga_mapping()
                    } else {
                        resource.mapping
                            .as_ref()
                            .expect("Error retrieving CPU memory mapping for host->guest")
                            .as_rutabaga_mapping()
                    };

                    // copy the resource into the buffer
                    let mut buffer = vec![0u8; resource_size as usize];
                    unsafe {
                        let src_ptr = res_rutabaga_mapping.ptr as *const u8;
                        ptr::copy_nonoverlapping(src_ptr, buffer.as_mut_ptr(), resource_size as usize);
                    }

                    // write the buffer to XDMA
                    self.xdma_device.write_bytes(DMA_ADDR, &mut buffer[..])?;
                    

                    let resp = kumquat_gpu_protocol_resp_host_copy_buffer {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_HOST_COPY_BUFFER,
                            ..Default::default()
                        },
                        copied: resource_size as u64,
                    };
                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }

                KumquatGpuProtocol::CopyFromCopyBuffer(cmd) => {
                    // guest->host: copy resource from XDMA

                    let resource_id = cmd.resource_id as u32;
                    let resource_size = cmd.resource_size;
                    let is_gpu_resource = cmd.is_gpu_resource;


                    let mut copied = 0;
                    if kumquat_gpu.resources.contains_key(&resource_id) {


                        let resource = kumquat_gpu
                            .resources
                            .get(&resource_id)
                            .ok_or(RutabagaError::InvalidResourceId)?;

                        let res_rutabaga_mapping = if is_gpu_resource == 1 {
                            resource.opt_mapping
                                .as_ref()
                                .expect("Error retrieving GPU memory mapping for guest->host")
                                .as_rutabaga_mapping()
                        } else {
                            resource.mapping
                                .as_ref()
                                .expect("Error retrieving CPU memory mapping for guest->host")
                                .as_rutabaga_mapping()
                        };

                        // read buffer from xdma 
                        let buffer = self.xdma_device.read_bytes(resource_size as usize, DMA_ADDR)?;

                        // copy buffer to resource
                        unsafe {
                            let copy_dest = res_rutabaga_mapping.ptr as *mut u8;
                            ptr::copy_nonoverlapping(buffer.as_ptr(), copy_dest, resource_size as usize);
                        }
                        
                        copied = resource_size;
                    } 
                    

                    // let guest know that copy is complete
                    let resp = kumquat_gpu_protocol_resp_host_copy_buffer {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_RESP_HOST_COPY_BUFFER,
                            ..Default::default()
                        },
                        copied: copied as u64,
                    };
                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?; 
                }

                KumquatGpuProtocol::SnapshotSave => {
                    kumquat_gpu.snapshot_buffer.set_position(0);
                    kumquat_gpu
                        .rutabaga
                        .snapshot(&mut kumquat_gpu.snapshot_buffer, SNAPSHOT_DIR)?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_OK_SNAPSHOT,
                        payload: 0,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::SnapshotRestore => {
                    kumquat_gpu
                        .rutabaga
                        .restore(&mut kumquat_gpu.snapshot_buffer, SNAPSHOT_DIR)?;

                    let resp = kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESP_OK_SNAPSHOT,
                        payload: 0,
                    };

                    self.stream.write(KumquatGpuProtocolWrite::Cmd(resp))?;
                }
                KumquatGpuProtocol::OkNoData => {
                    hung_up = true;
                }
                _ => {
                    error!("Unsupported protocol {:?}", protocol);
                    return Err(RutabagaError::Unsupported);
                }
            };
        }

        Ok(hung_up)
    }
}

impl AsFd for KumquatGpuConnection {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.stream.as_borrowed_file()
    }
}
