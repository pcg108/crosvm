// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::cmp::min;
use std::collections::BTreeMap as Map;
use std::convert::TryInto;
use std::path::PathBuf;
use std::slice::from_raw_parts_mut;
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::os::fd::RawFd;
use std::os::fd::AsRawFd;
use std::os::fd::FromRawFd;
use std::fs::File as StdFile;
use std::io::Read;
use std::fs::File;
use nix::unistd::{read, write};

use nix::poll::{poll, PollFd, PollFlags};
use nix::sys::mman::{mmap, MapFlags, ProtFlags};
use nix::sys::memfd::{memfd_create, MemFdCreateFlag};
use nix::unistd::{sysconf, SysconfVar};
use userfaultfd::{Event, Uffd, UffdBuilder, FaultKind};
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs;
use nix::fcntl::{open, OFlag};
use nix::sys::stat::Mode;
use std::os::fd::OwnedFd;

use std::ptr;
use libc::PROT_READ;
use libc::MAP_SHARED;
use std::num::NonZeroUsize;
use std::os::fd::AsFd;

use once_cell::sync::Lazy;

use rutabaga_gfx::kumquat_support::kumquat_gpu_protocol::*;
use rutabaga_gfx::kumquat_support::RutabagaEvent;
use rutabaga_gfx::kumquat_support::RutabagaMemoryMapping;
use rutabaga_gfx::kumquat_support::RutabagaReader;
use rutabaga_gfx::kumquat_support::RutabagaSharedMemory;
use rutabaga_gfx::kumquat_support::RutabagaStream;
use rutabaga_gfx::kumquat_support::RutabagaTube;
use rutabaga_gfx::kumquat_support::RutabagaTubeType;
use rutabaga_gfx::kumquat_support::RutabagaWriter;
use rutabaga_gfx::RutabagaDescriptor;
use rutabaga_gfx::RutabagaError;
use rutabaga_gfx::RutabagaGralloc;
use rutabaga_gfx::RutabagaGrallocBackendFlags;
use rutabaga_gfx::RutabagaHandle;
use rutabaga_gfx::RutabagaIntoRawDescriptor;
use rutabaga_gfx::RutabagaMappedRegion;
use rutabaga_gfx::RutabagaMapping;
use rutabaga_gfx::RutabagaRawDescriptor;
use rutabaga_gfx::RutabagaResult;
use rutabaga_gfx::VulkanInfo;
use rutabaga_gfx::RUTABAGA_FLAG_FENCE;
use rutabaga_gfx::RUTABAGA_FLAG_FENCE_HOST_SHAREABLE;
use rutabaga_gfx::RUTABAGA_FLAG_INFO_RING_IDX;
use rutabaga_gfx::RUTABAGA_MAP_ACCESS_RW;
use rutabaga_gfx::RUTABAGA_MAP_CACHE_CACHED;
use rutabaga_gfx::RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD;
use crate::rutabaga_gfx::AsRawDescriptor;

use crate::virtgpu::defines::*;
use crate::rutabaga_gfx::RutabagaFromRawDescriptor;

pub const RUTABAGA_MEM_HANDLE_TYPE_LOCAL_FD: u32 = 0x0011;

const VK_ICD_FILENAMES: &str = "VK_ICD_FILENAMES";

// The Tesla V-100 driver seems to enter a power management mode and stops being available to the
// Vulkan loader if more than a certain number of VK instances are created in the same process.
//
// This behavior is reproducible via:
//
// GfxstreamEnd2EndTests --gtest_filter="*MultiThreadedVkMapMemory*"
//
// Workaround this by having a singleton gralloc per-process.
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

fn copy_resource_from_local_to_host(resource_handle: &RutabagaHandle, size: u64, connection_id: u64) -> Result<(), Box<dyn std::error::Error>> {

    // println!("  guest->host: Copying handle: {:?} into copy-buffer", resource_handle);

    let resource_addr = unsafe {
        mmap(
            None,
            NonZeroUsize::new(size as usize).unwrap(),
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            Some(resource_handle.os_handle.as_fd()),
            0, // Offset
        )
    };

    let mut buffer = vec![0u8; size as usize];
    unsafe {
        let src_ptr = match resource_addr {
            Ok(addr) => {
                ptr::copy_nonoverlapping(addr as *const u8, buffer.as_mut_ptr(), size as usize);
            }
            Err(err) => {
                eprintln!("Error obtaining resource address: {}", err);
            }
        };
    }


    let file_path = format!("{}{}", "/tmp/copy-buffer-", connection_id);
    let c_file_path = CString::new(file_path).unwrap();
    let cstr_file_path = c_file_path.as_c_str();

    let raw_fd = open(
        cstr_file_path,
        OFlag::O_RDWR,
        Mode::S_IRUSR | Mode::S_IWUSR,
    ).expect("Error opening copy-buffer");
    let file = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    // Map the file into memory
    let copy_buffer_addr = unsafe {
        mmap(
            None,
            NonZeroUsize::new(size as usize).unwrap(),
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            Some(file.as_fd()),
            0,
        ).expect("error with mmap")
    };

    unsafe {
        let dest_ptr = copy_buffer_addr as *mut u8;
        ptr::copy_nonoverlapping(buffer.as_mut_ptr(), dest_ptr, size as usize);
    }

    Ok(())

}

fn copy_resource_from_host_to_local(resource_handle: &RutabagaHandle, size: u64, connection_id: u64) -> Result<(), Box<dyn std::error::Error>>{

    // println!("  host->guest: Copying copy-buffer into handle: {:?}", resource_handle);

    let file_path = format!("{}{}", "/tmp/copy-buffer-", connection_id);
    let c_file_path = CString::new(file_path).unwrap();
    let cstr_file_path = c_file_path.as_c_str();

    let raw_fd = open(
        cstr_file_path,
        OFlag::O_RDWR,
        Mode::S_IRUSR | Mode::S_IWUSR,
    ).expect("Error opening copy-buffer");
    let file = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    // Map the file into memory
    let copy_buffer_addr = unsafe {
        mmap(
            None,
            NonZeroUsize::new(size as usize).unwrap(),
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            Some(file.as_fd()),
            0,
        ).expect("error with mmap")
    };
    

    let mut buffer = vec![0u8; size as usize];
    unsafe {
        let src_ptr = copy_buffer_addr as *const u8;
        ptr::copy_nonoverlapping(src_ptr, buffer.as_mut_ptr(), size as usize);
    }

    let resource_addr = unsafe {
        mmap(
            None,
            NonZeroUsize::new(size as usize).unwrap(),
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            Some(resource_handle.os_handle.as_fd()),
            0, // Offset
        )
    };

    unsafe {
        let dest_ptr = match resource_addr {
            Ok(addr) => {
                ptr::copy_nonoverlapping(buffer.as_mut_ptr(), addr as *mut u8, size as usize);
            }
            Err(err) => {
                eprintln!("Error obtaining resource address: {}", err);
            }
        };
    }

    Ok(())

}


// UFFD Handler
fn fault_handler_thread(shared_uffd: Arc<Uffd>, 
    addr_to_resource: Arc<Mutex<HashMap<(usize, usize), u32>>>, 
    stream_lock: Arc<Mutex<RutabagaStream>>,
    connection_id: u64) {

    let uffd_fd: RawFd = shared_uffd.as_raw_fd();
    let file = unsafe { StdFile::from_raw_fd(uffd_fd) };

    let page_size = sysconf(SysconfVar::PAGE_SIZE).unwrap().unwrap() as usize;

    // let page = unsafe {
    //     mmap(
    //         None,
    //         page_size.try_into().unwrap(),
    //         ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
    //         MapFlags::MAP_PRIVATE | MapFlags::MAP_ANONYMOUS,
    //         None::<std::os::fd::BorrowedFd>,
    //         0,
    //     )
    //     .expect("mmap")
    // };

    // Loop, handling incoming events on the userfaultfd file descriptor
    let mut _fault_cnt = 0;
    loop {
        // See what poll() tells us about the userfaultfd

        let mut fds = [PollFd::new(&file, PollFlags::POLLIN)];
        let nready = poll(&mut fds, -1).expect("poll");
        let pollfd = fds[0];

        // println!("\nfault_handler_thread():");
        let revents = pollfd.revents().unwrap();
        // println!(
        //     "    poll() returns: nready = {}; POLLIN = {}; POLLERR = {}",
        //     nready,
        //     revents.contains(PollFlags::POLLIN),
        //     revents.contains(PollFlags::POLLERR),
        // );

        // Read an event from the userfaultfd
        let event = shared_uffd
            .read_event()
            .expect("read uffd_msg")
            .expect("uffd_msg ready");

        if let Event::Pagefault { kind, addr, .. } = event {
            // Display info about the page-fault event

            println!("    UFFD_EVENT_PAGEFAULT event: {:?}", event);

            match kind {
                FaultKind::Missing => {
                    let mut resource_id = 0;
                    let mut resource_size = 0;
                {
                    // search map keys for the resource corresponding to the faulting address
                    let a2r_map = addr_to_resource.lock().unwrap();
                    for &(blob_start_addr, blob_size) in a2r_map.keys() {
                        if blob_start_addr <= addr as usize && addr as usize <= blob_start_addr + blob_size {
                            if let Some(&value) = a2r_map.get(&(blob_start_addr, blob_size)) {
                                resource_id = value;
                                resource_size = blob_size;
                            } else {
                                // println!("Key not found");
                            }
                            break;
                        }
                    }
                    if resource_id == 0 { panic!("Userfaultfd: could not find address in any regions"); }
                    else { println!("        found address in resource: {}", resource_id); }
                }

                let ibuffs_ids = GLOBAL_INSTR_BUFFERS.lock().unwrap();
                // send signal for host to copy resource-id to host copy-buffer
                let mut stream = stream_lock.lock().unwrap();
                let copy_into_copy_buffer = kumquat_gpu_protocol_host_copy_into_copy_buffer {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_HOST_COPY_INTO_COPY_BUFFER,
                        ..Default::default()
                    },
                    resource_id: resource_id as u64,
                    resource_size: resource_size as u64,
                    is_gpu_resource: if ibuffs_ids.contains(&resource_id) { 0 } else { 1 }
                };
                let result = stream.write(KumquatGpuProtocolWrite::Cmd(copy_into_copy_buffer));
                match result {
                    Ok(_) => {}
                    Err(e) => {
                        println!("Error writing to stream: {}", e);
                    }
                }

                // once stream responds with confirmation, copy from the host-copy buffer to the missing resource ID
                let protocols = stream.read();
                let _ = match protocols.expect("Error reading from stream").remove(0) {
                    KumquatGpuProtocol::RespCopyCopyBuffer(resp) => {
                        if resp.copied <= 0 {
                            panic!("unsuccessful host copy into copy buffer");
                        }

                        // get the handle to the resource
                        let r2h_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap();
                        let r_id = resource_id as u32;
                        let rutabaga_handle = r2h_map.get(&r_id);

                        match rutabaga_handle {
                            Some((handle, _size)) => {
                                let _ = copy_resource_from_host_to_local(handle, resp.copied, connection_id);
                            }
                            None => {
                                // Handle the case where `rutabaga_handle` is `None`
                                eprintln!("Rutabaga handle is missing");
                            }
                        }
                    }
                    _ => {
                        println!("Error RespCopyIntoCopyBuffer");
                    }
                };

                // wake up faulting thread
                let _ = shared_uffd.wake(addr, page_size);
                
                // for c in unsafe { std::slice::from_raw_parts_mut(page as *mut u8, page_size) } {
                //     *c = b'A' + fault_cnt % 20;
                // }
                // // fault_cnt += 1;

                // let dst = (addr as usize & !(page_size - 1)) as *mut c_void;
                // let copy = unsafe { shared_uffd.copy(page, dst, page_size, true).expect("uffd copy") };

                // println!("        (uffdio_copy.copy returned {})", copy);
                }
            }

        } else {
            panic!("Unexpected event on userfaultfd");
        }

    }
}


static GLOBAL_ID_HANDLE_MAP: Lazy<Arc<Mutex<HashMap<u32, (Arc<RutabagaHandle>, usize)>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(HashMap::new()))
});

static GLOBAL_INSTR_BUFFERS: Lazy<Arc<Mutex<HashSet<u32>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(HashSet::new()))
});

pub struct VirtGpuResource {
    resource_id: u32,
    size: usize,
    handle: Arc<RutabagaHandle>,
    attached_fences: Vec<RutabagaHandle>,
    vulkan_info: VulkanInfo,
    system_mapping: Option<RutabagaMemoryMapping>,
    gpu_mapping: Option<Box<dyn RutabagaMappedRegion>>,
}

impl VirtGpuResource {
    pub fn new(
        resource_id: u32,
        size: usize,
        handle: Arc<RutabagaHandle>,
        vulkan_info: VulkanInfo,
    ) -> VirtGpuResource {
        VirtGpuResource {
            resource_id,
            size,
            handle,
            attached_fences: Vec::new(),
            vulkan_info,
            system_mapping: None,
            gpu_mapping: None,
        }
    }
}

pub struct VirtGpuKumquat {
    context_id: u32,
    connection_id: u64,
    id_allocator: u32,
    capset_mask: u64,
    stream_lock: Arc<Mutex<RutabagaStream>>,
    capsets: Map<u32, Vec<u8>>,
    resources: Map<u32, VirtGpuResource>,
    shared_uffd: Arc<Uffd>,
    address_to_resource_id: Arc<Mutex<HashMap<(usize, usize), u32>>>,
    stream_resource_id: u32,
    stream_resource_size: u32,
}

fn print_open_file_descriptors() {
    let pid = std::process::id(); // Get the current process ID
    let fd_path = format!("/proc/{}/fd", pid); // Path to the fd directory

    match fs::read_dir(&fd_path) {
        Ok(entries) => {
            println!("Open file descriptors for PID {}:", pid);
            for entry in entries {
                if let Ok(entry) = entry {
                    let fd = entry.file_name();
                    let fd_path: PathBuf = entry.path();

                    // Try to resolve the symbolic link
                    match fs::read_link(&fd_path) {
                        Ok(target) => println!("FD {} -> {}", fd.to_string_lossy(), target.display()),
                        Err(err) => println!("FD {} -> (error: {})", fd.to_string_lossy(), err),
                    }
                }
            }
        }
        Err(err) => {
            eprintln!("Failed to read /proc/{}/fd: {}", pid, err);
        }
    }
}



impl VirtGpuKumquat {
    pub fn new(gpu_socket: &str) -> RutabagaResult<VirtGpuKumquat> {

        println!("New Virt GPU Kumquat");

        let path = PathBuf::from(gpu_socket);
        println!("connected to: {:?}", path);
        let connection = RutabagaTube::new(path, RutabagaTubeType::Packet)?;
        let mut capset_mask = 0;
        let mut capsets: Map<u32, Vec<u8>> = Default::default();
        let mut connection_id = 0;

        let stream_lock = Arc::new(Mutex::new(RutabagaStream::new(connection)));


        {
            let mut stream = stream_lock.lock().unwrap();

            let get_conn_id = kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_GET_CONNECTION_ID,
                ..Default::default()
            };

            stream.write(KumquatGpuProtocolWrite::Cmd(get_conn_id))?;
            let mut protocols = stream.read()?;
            connection_id = match protocols.remove(0) {
                KumquatGpuProtocol::RespConnectionId(num) => num,
                _ => return Err(RutabagaError::Unsupported),
            };


            let get_num_capsets = kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_GET_NUM_CAPSETS,
                ..Default::default()
            };

            stream.write(KumquatGpuProtocolWrite::Cmd(get_num_capsets))?;
            let mut protocols = stream.read()?;
            let num_capsets = match protocols.remove(0) {
                KumquatGpuProtocol::RespNumCapsets(num) => num,
                _ => return Err(RutabagaError::Unsupported),
            };

            for capset_index in 0..num_capsets {
                let get_capset_info = kumquat_gpu_protocol_ctrl_hdr {
                    type_: KUMQUAT_GPU_PROTOCOL_GET_CAPSET_INFO,
                    payload: capset_index,
                };

                stream.write(KumquatGpuProtocolWrite::Cmd(get_capset_info))?;
                protocols = stream.read()?;
                let resp_capset_info = match protocols.remove(0) {
                    KumquatGpuProtocol::RespCapsetInfo(info) => info,
                    _ => return Err(RutabagaError::Unsupported),
                };

                let get_capset = kumquat_gpu_protocol_get_capset {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_GET_CAPSET,
                        ..Default::default()
                    },
                    capset_id: resp_capset_info.capset_id,
                    capset_version: resp_capset_info.version,
                };

                stream.write(KumquatGpuProtocolWrite::Cmd(get_capset))?;
                protocols = stream.read()?;
                let capset = match protocols.remove(0) {
                    KumquatGpuProtocol::RespCapset(capset) => capset,
                    _ => return Err(RutabagaError::Unsupported),
                };

                capset_mask = 1u64 << resp_capset_info.capset_id | capset_mask;
                capsets.insert(resp_capset_info.capset_id, capset);
            }
            
        }

        let uffd = UffdBuilder::new()
                    .close_on_exec(true)
                    .non_blocking(true)
                    .user_mode_only(true)
                    .create()
                    .expect("uffd creation");
        let shared_uffd = Arc::new(uffd);

        // map the address of the local resource to its resource ID (used in page fault handler)
        let addr_to_resource: HashMap<(usize, usize), u32> = HashMap::new();
        let address_to_resource_id = Arc::new(Mutex::new(addr_to_resource));

        // map the resource ID to its rutabaga handle
        let res_id_to_res_handle: HashMap<u32, (Arc<RutabagaHandle>, usize)> = Default::default();
        
        // Create a thread that will process the userfaultfd events
        let cloned_ptr = Arc::clone(&shared_uffd);
        let cloned_a2r = Arc::clone(&address_to_resource_id);
        let cloned_stream = Arc::clone(&stream_lock);
        let _s = std::thread::spawn(move || fault_handler_thread(cloned_ptr, cloned_a2r, cloned_stream, connection_id));

        Ok(VirtGpuKumquat {
            context_id: 0,
            connection_id: connection_id,
            id_allocator: 0,
            capset_mask,
            stream_lock,
            capsets,
            resources: Default::default(),
            shared_uffd,
            address_to_resource_id,
            stream_resource_id: 0,
            stream_resource_size: 0,
        })
    }

    pub fn allocate_id(&mut self) -> u32 {
        self.id_allocator = self.id_allocator + 1;
        self.id_allocator
    }

    pub fn get_param(&self, getparam: &mut VirtGpuParam) -> RutabagaResult<()> {
        getparam.value = match getparam.param {
            VIRTGPU_KUMQUAT_PARAM_3D_FEATURES => (self.capset_mask != 0) as u64,
            VIRTGPU_KUMQUAT_PARAM_CAPSET_QUERY_FIX..=VIRTGPU_KUMQUAT_PARAM_CONTEXT_INIT => 1,
            VIRTGPU_KUMQUAT_PARAM_SUPPORTED_CAPSET_IDS => self.capset_mask,
            VIRTGPU_KUMQUAT_PARAM_EXPLICIT_DEBUG_NAME => 0,
            VIRTGPU_KUMQUAT_PARAM_FENCE_PASSING => 1,
            _ => return Err(RutabagaError::Unsupported),
        };

        Ok(())
    }

    pub fn get_contextid(&self, getcontextid: &mut VirtGpuContextId) -> RutabagaResult<()> {
        getcontextid.context_id = self.context_id;
        Ok(())
    }

    pub fn get_caps(&self, capset_id: u32, slice: &mut [u8]) -> RutabagaResult<()> {
        let caps = self
            .capsets
            .get(&capset_id)
            .ok_or(RutabagaError::InvalidCapset)?;
        let length = min(slice.len(), caps.len());
        slice.copy_from_slice(&caps[0..length]);
        Ok(())
    }

    pub fn context_create(&mut self, capset_id: u64, name: &str) -> RutabagaResult<u32> {
        let mut stream = self.stream_lock.lock().unwrap();

        let mut debug_name = [0u8; 64];
        debug_name
            .iter_mut()
            .zip(name.bytes())
            .for_each(|(dst, src)| *dst = src);

        let context_create = kumquat_gpu_protocol_ctx_create {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_CTX_CREATE,
                ..Default::default()
            },
            nlen: 0,
            context_init: capset_id.try_into()?,
            debug_name,
        };

        stream.write(KumquatGpuProtocolWrite::Cmd(context_create))?;
        let mut protocols = stream.read()?;
        self.context_id = match protocols.remove(0) {
            KumquatGpuProtocol::RespContextCreate(ctx_id) => ctx_id,
            _ => return Err(RutabagaError::Unsupported),
        };

        Ok(self.context_id)
    }

    pub fn resource_create_3d(
        &mut self,
        create_3d: &mut VirtGpuResourceCreate3D,
    ) -> RutabagaResult<()> {

        // create a memfd_create backed region locally for this instruction stream
        let c_string = CString::new("instruction stream memory").expect("Failed to create name");
        let local_fd = memfd_create(c_string.as_c_str(),MemFdCreateFlag::MFD_CLOEXEC | MemFdCreateFlag::MFD_ALLOW_SEALING).expect("Failed to create memfd");
        nix::unistd::ftruncate(&local_fd, create_3d.size as i64).expect("Failed to resize memfd");

        let resource_create_3d = kumquat_gpu_protocol_resource_create_3d {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_3D,
                ..Default::default()
            },
            target: create_3d.target,
            format: create_3d.format,
            bind: create_3d.bind,
            width: create_3d.width,
            height: create_3d.height,
            depth: create_3d.depth,
            array_size: create_3d.array_size,
            last_level: create_3d.last_level,
            nr_samples: create_3d.nr_samples,
            flags: create_3d.flags,
            size: create_3d.size,
            stride: create_3d.stride,
            ctx_id: self.context_id,
        };

        let mut stream = self.stream_lock.lock().unwrap();

        stream
            .write(KumquatGpuProtocolWrite::Cmd(resource_create_3d))?;
        let mut protocols = stream.read()?;
        let resource = match protocols.remove(0) {
            KumquatGpuProtocol::RespResourceCreate(resp, _handle) => {
                let size: usize = create_3d.size.try_into()?;
                // let arc_handle = Arc::new(_handle);

                // create a VirtGpuResource using the local stream, rather than the handle coming back from the host
                let t_handle = unsafe { RutabagaDescriptor::from_raw_descriptor(local_fd.as_raw_fd()) };
                let os_handle = t_handle.try_clone()?;

                let local_handle = RutabagaHandle {
                    os_handle,
                    handle_type: RUTABAGA_MEM_HANDLE_TYPE_LOCAL_FD,
                };

                // wrap in Arc to allow usage in fault handler
                let arc_handle = Arc::new(local_handle);

                VirtGpuResource::new(resp.resource_id, size, arc_handle, resp.vulkan_info)
            }
            _ => return Err(RutabagaError::Unsupported),
        };
        drop(stream); // have to release to call self.<...>


        self.stream_resource_id = resource.resource_id;
        self.stream_resource_size = create_3d.size;

        let mut ibuffs_ids = GLOBAL_INSTR_BUFFERS.lock().unwrap();
        ibuffs_ids.insert(resource.resource_id);

        // map resource-id to the rutabaga-handle for the resource
        // insert into global static map
        let mut global_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap(); 
        let clone = Arc::clone(&resource.handle);
        println!("inserting stream handle into map: {}: {:?}", resource.resource_id, clone);
        global_map.insert(resource.resource_id, (clone, create_3d.size as usize));


        create_3d.res_handle = resource.resource_id;
        create_3d.bo_handle = self.allocate_id();
        self.resources.insert(create_3d.bo_handle, resource);

        Ok(())
    }

    pub fn resource_create_blob(
        &mut self,
        create_blob: &mut VirtGpuResourceCreateBlob,
        blob_cmd: &[u8],
    ) -> RutabagaResult<()> {
        let mut stream = self.stream_lock.lock().unwrap();

        // this doesn't seem to be getting called since LINUX_GUEST_BUILD is not defined
        if blob_cmd.len() != 0 { 

            let submit_command = kumquat_gpu_protocol_cmd_submit {
                hdr: kumquat_gpu_protocol_ctrl_hdr {
                    type_: KUMQUAT_GPU_PROTOCOL_SUBMIT_3D,
                    ..Default::default()
                },
                ctx_id: self.context_id,
                pad: 0,
                size: blob_cmd.len().try_into()?,
                num_in_fences: 0,
                flags: 0,
                ring_idx: 0,
                padding: Default::default(),
            };

            let mut data: Vec<u8> = vec![0; blob_cmd.len()];
            data.copy_from_slice(blob_cmd);

            stream
                .write(KumquatGpuProtocolWrite::CmdWithData(submit_command, data))?;
        }

        // create a memfd_create backed region locally for this blob
        let c_string = CString::new("blob memfd").expect("Failed to create name");
        let local_fd = memfd_create(c_string.as_c_str(),MemFdCreateFlag::MFD_CLOEXEC | MemFdCreateFlag::MFD_ALLOW_SEALING).expect("Failed to create memfd");
        nix::unistd::ftruncate(&local_fd, create_blob.size as i64).expect("Failed to resize memfd");
        
        
        // call the host to create a blob 
        let resource_create_blob = kumquat_gpu_protocol_resource_create_blob {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_BLOB,
                ..Default::default()
            },
            ctx_id: self.context_id,
            blob_mem: create_blob.blob_mem,
            blob_flags: create_blob.blob_flags,
            padding: 0,
            blob_id: create_blob.blob_id,
            size: create_blob.size,
        };

        stream
            .write(KumquatGpuProtocolWrite::Cmd(resource_create_blob))?;
        let mut protocols = stream.read()?;
        let resource = match protocols.remove(0) {
            KumquatGpuProtocol::RespResourceCreate(resp, _handle) => {
                let size: usize = create_blob.size.try_into()?;
                // let handle = Arc::new(_handle);
                // VirtGpuResource::new(resp.resource_id, size, handle, resp.vulkan_info)

                // create a VirtGpuResource using the local blob, rather than the handle coming back from the host
                let t_handle = unsafe { RutabagaDescriptor::from_raw_descriptor(local_fd.as_raw_fd()) };
                let os_handle = t_handle.try_clone()?;

                let local_handle = RutabagaHandle {
                    os_handle,
                    handle_type: RUTABAGA_MEM_HANDLE_TYPE_LOCAL_FD,
                };

                // wrap in Arc to allow usage in fault handler
                let arc_handle = Arc::new(local_handle);

                VirtGpuResource::new(resp.resource_id, size, arc_handle, resp.vulkan_info)
            }
            _ => {
                return Err(RutabagaError::Unsupported);
            }
        };
        drop(stream); 

        // map resource-id to the rutabaga-handle for the resource
        // insert into global static map
        let mut global_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap(); 
        let clone = Arc::clone(&resource.handle);
        println!("inserting into map: {}: {:?}", resource.resource_id, clone);
        global_map.insert(resource.resource_id, (clone, create_blob.size as usize));

            
        println!("created local guest handle: {:?}, size: {:?}", resource.handle, resource.size) ;
        create_blob.res_handle = resource.resource_id;
        create_blob.bo_handle = self.allocate_id();
        self.resources.insert(create_blob.bo_handle, resource);

        Ok(())
    }

    pub fn resource_unref(&mut self, bo_handle: u32) -> RutabagaResult<()> {
        let resource = self
            .resources
            .remove(&bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        let detach_resource = kumquat_gpu_protocol_ctx_resource {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_CTX_DETACH_RESOURCE,
                ..Default::default()
            },
            ctx_id: self.context_id,
            resource_id: resource.resource_id,
        };

        let mut stream = self.stream_lock.lock().unwrap();
        stream.write(KumquatGpuProtocolWrite::Cmd(detach_resource))?;

        // also need to remove the resource from local maps
        let mut addr_to_resource_id = self.address_to_resource_id.lock().unwrap();
        
        let key = resource.resource_id;
        println!("  removing resource-id: {}", key);

        let mut global_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap(); 
        global_map.remove(&key);

        let mut key_to_remove_2: (usize, usize) = Default::default();
        for &(addr, size) in addr_to_resource_id.keys() {
            if let Some(&res_id) = addr_to_resource_id.get(&(addr, size)) {
                if res_id == key {
                    key_to_remove_2 = (addr, size);
                    break;
                }
            } else {}
        }
        println!("  removing (addr, size): ({:?}, {})", key_to_remove_2.0 as *const u8, key_to_remove_2.1);
        addr_to_resource_id.remove(&key_to_remove_2);


        Ok(())
    }

    pub fn map(&mut self, bo_handle: u32) -> RutabagaResult<RutabagaMapping> {

        let resource = self
            .resources
            .get_mut(&bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        if let Some(ref system_mapping) = resource.system_mapping {
            let rutabaga_mapping = system_mapping.as_rutabaga_mapping();
            Ok(rutabaga_mapping)
        } else if let Some(ref gpu_mapping) = resource.gpu_mapping {
            let rutabaga_mapping = gpu_mapping.as_rutabaga_mapping();
            Ok(rutabaga_mapping)
        } else {
            println!("mapped host->guest handle: {:?}, size: {:?}", resource.handle, resource.size) ;
            let clone = resource.handle.try_clone()?;

            if clone.handle_type == RUTABAGA_MEM_HANDLE_TYPE_LOCAL_FD {
                let mapping = RutabagaMemoryMapping::from_safe_descriptor(
                    clone.os_handle,
                    resource.size,
                    RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
                )?;

                // Register the memory range of the mapping we just created for handling by the userfaultfd
                // object. In mode, we request to track missing pages (i.e., pages that have not yet been
                // faulted in) 
                let addr = mapping.mapping.addr.as_ptr();
                println!("  addr: {:?}", addr);

                // register with UFFD if this isn't the instruction stream (we explicitly copy this)
                if (resource.resource_id != self.stream_resource_id) {
                    self.shared_uffd.register(addr, resource.size as usize).expect("uffd.register()");
                }
                
                // map the address of the resource to its ID, for use in page fault handler
                let mut map = self.address_to_resource_id.lock().unwrap();
                map.insert((addr as usize, resource.size as usize), resource.resource_id);

            
                let rutabaga_mapping = mapping.as_rutabaga_mapping();
                resource.system_mapping = Some(mapping);
                Ok(rutabaga_mapping)
            } else if clone.handle_type == RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD {
                let region = gralloc().lock().unwrap().import_and_map(
                    clone,
                    resource.vulkan_info,
                    resource.size as u64,
                )?;

                let rutabaga_mapping = region.as_rutabaga_mapping();
                println!("  gpu addr: {:?}", rutabaga_mapping.ptr as *mut u8);
                resource.gpu_mapping = Some(region);
                Ok(rutabaga_mapping)
            } else {
                let mapping = RutabagaMemoryMapping::from_safe_descriptor(
                    clone.os_handle,
                    resource.size,
                    RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
                )?;

                let addr = mapping.mapping.addr.as_ptr();
                println!("  addr: {:?}", addr);

                let rutabaga_mapping = mapping.as_rutabaga_mapping();
                resource.system_mapping = Some(mapping);
                Ok(rutabaga_mapping)
            }
        }
    }

    pub fn copy_resources_to_host(&mut self) -> RutabagaResult<u32> {
        // return Ok(0);
        // println!("Copying all resources back to host:");
        
        let mut stream = self.stream_lock.lock().unwrap();

        let mut bytes_copied = 0;
        let mut keys_to_remove: Vec<u32> = Vec::new();

        let ibuffs_ids = GLOBAL_INSTR_BUFFERS.lock().unwrap();
        let mut global_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap(); 
        for &resource_id in global_map.keys() {

            if (ibuffs_ids.contains(&resource_id)) {continue;}
            
            println!("  sending resource {}", resource_id);

            match global_map.get(&resource_id) {
                Some((resource_handle, resource_size)) => {
                    // copy from local to host copy-buffer
                    let _  = copy_resource_from_local_to_host(resource_handle, *resource_size as u64, self.connection_id);

                    // send signal for host to copy copy-buffer into resource 
                    let host_copy_from_copy_buffer = kumquat_gpu_protocol_host_copy_from_copy_buffer {
                        hdr: kumquat_gpu_protocol_ctrl_hdr {
                            type_: KUMQUAT_GPU_PROTOCOL_HOST_COPY_FROM_COPY_BUFFER,
                            ..Default::default()
                        },
                        resource_id: resource_id as u64,
                        resource_size: *resource_size as u64,
                        is_gpu_resource: 1,
                    };

                    stream
                        .write(KumquatGpuProtocolWrite::Cmd(host_copy_from_copy_buffer))?;
                    let mut protocols = stream.read()?;
                    let bytes = match protocols.remove(0) {
                        KumquatGpuProtocol::RespCopyCopyBuffer(resp) => {
                            // println!("  Host copied {} bytes from copy-buffer", resp.copied);
                            resp.copied
                        }
                        _ => {
                            return Err(RutabagaError::Unsupported);
                        }
                    };

                    if bytes == 0 {
                        keys_to_remove.push(resource_id);
                    }

                    bytes_copied += bytes;
                }
                None => {
                    println!("resource ID: {} not found in ID:handle map", resource_id);
                }
            }
        }

        let mut addr_to_resource_id = self.address_to_resource_id.lock().unwrap();
        let mut keys_to_remove_2: Vec<(usize, usize)> = Vec::new();
        for key in keys_to_remove {
            println!("  removing resource-id: {}", key);
            global_map.remove(&key);

            for &(addr, size) in addr_to_resource_id.keys() {
                if let Some(&res_id) = addr_to_resource_id.get(&(addr, size)) {
                    if res_id == key {
                        keys_to_remove_2.push((addr, size));
                    }
                } else {}
            }
        }

        for key in keys_to_remove_2 {
            println!("  removing (addr, size): ({:?}, {})", key.0 as *const u8, key.1);
            addr_to_resource_id.remove(&key);
        }
    

        return Ok(bytes_copied as u32);
    }

    pub fn unmap(&mut self, bo_handle: u32) -> RutabagaResult<()> {
        let resource = self
            .resources
            .get_mut(&bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        resource.system_mapping = None;
        resource.gpu_mapping = None;
        Ok(())
    }

    pub fn transfer_to_host(&mut self, transfer: &VirtGpuTransfer) -> RutabagaResult<()> {

        // copy instruction stream over to host before doing anything else 

        let mut stream = self.stream_lock.lock().unwrap();

        let mut global_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap(); 
        let resource_id = self.stream_resource_id;
        let mut bytes_copied = 0;
        match global_map.get(&resource_id) {
            Some((resource_handle, resource_size)) => {
                // copy from local to host copy-buffer
                let _  = copy_resource_from_local_to_host(resource_handle, *resource_size as u64, self.connection_id);

                // send signal for host to copy copy-buffer into resource 
                let host_copy_from_copy_buffer = kumquat_gpu_protocol_host_copy_from_copy_buffer {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_HOST_COPY_FROM_COPY_BUFFER,
                        ..Default::default()
                    },
                    resource_id: resource_id as u64,
                    resource_size: *resource_size as u64,
                    is_gpu_resource: 0,
                };

                stream
                    .write(KumquatGpuProtocolWrite::Cmd(host_copy_from_copy_buffer))?;
                let mut protocols = stream.read()?;
                let bytes = match protocols.remove(0) {
                    KumquatGpuProtocol::RespCopyCopyBuffer(resp) => {
                        // println!("  Host copied {} bytes from copy-buffer", resp.copied);
                        resp.copied
                    }
                    _ => {
                        return Err(RutabagaError::Unsupported);
                    }
                };

                bytes_copied += bytes;
            }
            None => {
                println!("resource ID: {} not found in ID:handle map", resource_id);
            }
        }

        if (bytes_copied < 0) {
            println!("copied 0 bytes of instruction buffer");
        }

        ////////////////////////////////


        let resource = self
            .resources
            .get_mut(&transfer.bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        let event = RutabagaEvent::new()?;
        let emulated_fence: RutabagaHandle = event.into();

        resource.attached_fences.push(emulated_fence.try_clone()?);

        let transfer_to_host = kumquat_gpu_protocol_transfer_host_3d {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_TRANSFER_TO_HOST_3D,
                ..Default::default()
            },
            box_: kumquat_gpu_protocol_box {
                x: transfer._box.x,
                y: transfer._box.y,
                z: transfer._box.z,
                w: transfer._box.w,
                h: transfer._box.h,
                d: transfer._box.d,
            },
            offset: transfer.offset,
            level: transfer.level,
            stride: transfer.stride,
            layer_stride: transfer.layer_stride,
            ctx_id: self.context_id,
            resource_id: resource.resource_id,
            padding: 0,
        };

        // let mut stream = self.stream_lock.lock().unwrap();
        
        stream.write(KumquatGpuProtocolWrite::CmdWithHandle(
            transfer_to_host,
            emulated_fence,
        ))?;
        
        Ok(())
    }

    pub fn transfer_from_host(&mut self, transfer: &VirtGpuTransfer) -> RutabagaResult<()> {

        let mut stream = self.stream_lock.lock().unwrap();

        let resource = self
            .resources
            .get_mut(&transfer.bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        let event = RutabagaEvent::new()?;
        let emulated_fence: RutabagaHandle = event.into();

        resource.attached_fences.push(emulated_fence.try_clone()?);
        let transfer_from_host = kumquat_gpu_protocol_transfer_host_3d {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_TRANSFER_FROM_HOST_3D,
                ..Default::default()
            },
            box_: kumquat_gpu_protocol_box {
                x: transfer._box.x,
                y: transfer._box.y,
                z: transfer._box.z,
                w: transfer._box.w,
                h: transfer._box.h,
                d: transfer._box.d,
            },
            offset: transfer.offset,
            level: transfer.level,
            stride: transfer.stride,
            layer_stride: transfer.layer_stride,
            ctx_id: self.context_id,
            resource_id: resource.resource_id,
            padding: 0,
        };

        stream.write(KumquatGpuProtocolWrite::CmdWithHandle(
            transfer_from_host,
            emulated_fence,
        ))?;

        //////////////////////////////////
        
        // signal host to copy instruction buffer into copy buffer and then copy in before doing anything else
        // do this after calling host to do transfer from host because we need the host to do the transfer before we copy it into our local
        
        let resource_id = self.stream_resource_id;
        let resource_size = self.stream_resource_size;

        // send signal for host to copy resource-id to host copy-buffer
        let copy_into_copy_buffer = kumquat_gpu_protocol_host_copy_into_copy_buffer {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_HOST_COPY_INTO_COPY_BUFFER,
                ..Default::default()
            },
            resource_id: resource_id as u64,
            resource_size: resource_size as u64,
            is_gpu_resource: 0,
        };
        let result = stream.write(KumquatGpuProtocolWrite::Cmd(copy_into_copy_buffer));
        match result {
            Ok(_) => {}
            Err(e) => {
                println!("Error writing to stream: {}", e);
            }
        }

        // once stream responds with confirmation, copy from the host-copy buffer to the missing resource ID
        let protocols = stream.read();
        let _ = match protocols.expect("Error reading from stream").remove(0) {
            KumquatGpuProtocol::RespCopyCopyBuffer(resp) => {
                if resp.copied <= 0 {
                    panic!("unsuccessful host copy into copy buffer");
                }

                // get the handle to the resource
                let r2h_map = GLOBAL_ID_HANDLE_MAP.lock().unwrap();
                let r_id = resource_id as u32;
                let rutabaga_handle = r2h_map.get(&r_id);

                match rutabaga_handle {
                    Some((handle, _size)) => {
                        let _ = copy_resource_from_host_to_local(handle, resp.copied, self.connection_id);
                    }
                    None => {
                        // Handle the case where `rutabaga_handle` is `None`
                        eprintln!("Rutabaga handle is missing");
                    }
                }
            }
            _ => {
                println!("Error RespCopyIntoCopyBuffer");
            }
        };

        ///////////////////////////


        Ok(())
    }

    pub fn submit_command(
        &mut self,
        flags: u32,
        bo_handles: &[u32],
        cmd: &[u8],
        ring_idx: u32,
        in_fences: &[u64],
        raw_descriptor: &mut RutabagaRawDescriptor,
    ) -> RutabagaResult<()> {
        let mut fence_opt: Option<RutabagaHandle> = None;
        let mut data: Vec<u8> = vec![0; cmd.len()];
        let mut host_flags = 0;
        let mut stream = self.stream_lock.lock().unwrap();

        if flags & VIRTGPU_KUMQUAT_EXECBUF_RING_IDX != 0 {
            host_flags = RUTABAGA_FLAG_INFO_RING_IDX;
        }

        let need_fence =
            bo_handles.len() != 0 || (flags & VIRTGPU_KUMQUAT_EXECBUF_FENCE_FD_OUT) != 0;

        let actual_fence = (flags & VIRTGPU_KUMQUAT_EXECBUF_SHAREABLE_OUT) != 0
            && (flags & VIRTGPU_KUMQUAT_EXECBUF_FENCE_FD_OUT) != 0;

        // Should copy from in-fences when gfxstream supports it.
        data.copy_from_slice(cmd);

        if actual_fence {
            host_flags |= RUTABAGA_FLAG_FENCE_HOST_SHAREABLE;
            host_flags |= RUTABAGA_FLAG_FENCE;
        } else if need_fence {
            host_flags |= RUTABAGA_FLAG_FENCE;
        }

        let submit_command = kumquat_gpu_protocol_cmd_submit {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_SUBMIT_3D,
                ..Default::default()
            },
            ctx_id: self.context_id,
            pad: 0,
            size: cmd.len().try_into()?,
            num_in_fences: in_fences.len().try_into()?,
            flags: host_flags,
            ring_idx: ring_idx.try_into()?,
            padding: Default::default(),
        };

        if need_fence {
            stream
                .write(KumquatGpuProtocolWrite::CmdWithData(submit_command, data))?;

            let mut protocols = stream.read()?;
            let fence = match protocols.remove(0) {
                KumquatGpuProtocol::RespCmdSubmit3d(_fence_id, handle) => handle,
                _ => {
                    return Err(RutabagaError::Unsupported);
                }
            };

            for handle in bo_handles {
                // We could support implicit sync with real fences, but the need does not exist.
                if actual_fence {
                    return Err(RutabagaError::Unsupported);
                }

                let resource = self
                    .resources
                    .get_mut(handle)
                    .ok_or(RutabagaError::InvalidResourceId)?;

                resource.attached_fences.push(fence.try_clone()?);
            }

            fence_opt = Some(fence);
        } else {
            stream
                .write(KumquatGpuProtocolWrite::CmdWithData(submit_command, data))?;
        }

        if flags & VIRTGPU_KUMQUAT_EXECBUF_FENCE_FD_OUT != 0 {
            *raw_descriptor = fence_opt
                .ok_or(RutabagaError::SpecViolation("no fence found"))?
                .os_handle
                .into_raw_descriptor();
        }

        Ok(())
    }

    pub fn wait(&mut self, bo_handle: u32) -> RutabagaResult<()> {
        let resource = self
            .resources
            .get_mut(&bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        let new_fences: Vec<RutabagaHandle> = std::mem::take(&mut resource.attached_fences);
        for fence in new_fences {
            let event: RutabagaEvent = fence.try_into()?;
            event.wait()?;
        }

        Ok(())
    }

    pub fn resource_export(
        &mut self,
        bo_handle: u32,
        flags: u32,
    ) -> RutabagaResult<RutabagaHandle> {
        let resource = self
            .resources
            .get_mut(&bo_handle)
            .ok_or(RutabagaError::InvalidResourceId)?;

        if flags & VIRTGPU_KUMQUAT_EMULATED_EXPORT != 0 {
            let descriptor: RutabagaDescriptor =
                RutabagaSharedMemory::new("virtgpu_export", VIRTGPU_KUMQUAT_PAGE_SIZE as u64)?
                    .into();

            let clone = descriptor.try_clone()?;

            // Creating the mapping closes the cloned descriptor.
            let mapping = RutabagaMemoryMapping::from_safe_descriptor(
                clone,
                VIRTGPU_KUMQUAT_PAGE_SIZE,
                RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
            )?;
            let rutabaga_mapping = mapping.as_rutabaga_mapping();

            let mut slice: &mut [u8] = unsafe {
                from_raw_parts_mut(rutabaga_mapping.ptr as *mut u8, VIRTGPU_KUMQUAT_PAGE_SIZE)
            };
            let mut writer = RutabagaWriter::new(&mut slice);
            writer.write_obj(resource.resource_id)?;

            // Opaque to users of this API, shared memory internally
            Ok(RutabagaHandle {
                os_handle: descriptor,
                handle_type: RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD,
            })
        } else {
            let clone = resource.handle.try_clone()?;
            Ok(clone)
        }
    }

    pub fn resource_import(
        &mut self,
        handle: RutabagaHandle,
        bo_handle: &mut u32,
        resource_handle: &mut u32,
        size: &mut u64,
    ) -> RutabagaResult<()> {
        let mut stream = self.stream_lock.lock().unwrap();

        let clone = handle.try_clone()?;
        let mapping = RutabagaMemoryMapping::from_safe_descriptor(
            clone.os_handle,
            VIRTGPU_KUMQUAT_PAGE_SIZE,
            RUTABAGA_MAP_CACHE_CACHED | RUTABAGA_MAP_ACCESS_RW,
        )?;

        let rutabaga_mapping = mapping.as_rutabaga_mapping();

        let mut slice: &mut [u8] = unsafe {
            from_raw_parts_mut(rutabaga_mapping.ptr as *mut u8, VIRTGPU_KUMQUAT_PAGE_SIZE)
        };

        let mut reader = RutabagaReader::new(&mut slice);
        *resource_handle = reader.read_obj()?;

        let attach_resource = kumquat_gpu_protocol_ctx_resource {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_CTX_ATTACH_RESOURCE,
                ..Default::default()
            },
            ctx_id: self.context_id,
            resource_id: *resource_handle,
        };

        stream
            .write(KumquatGpuProtocolWrite::Cmd(attach_resource))?;
        let resource = VirtGpuResource::new(
            *resource_handle,
            VIRTGPU_KUMQUAT_PAGE_SIZE,
            Arc::new(handle),
            Default::default(),
        );
        drop(stream); // have to release to call self.<...>

        *bo_handle = self.allocate_id();
        // Should ask the server about the size long-term.
        *size = VIRTGPU_KUMQUAT_PAGE_SIZE as u64;
        self.resources.insert(*bo_handle, resource);

        Ok(())
    }

    pub fn snapshot(&mut self) -> RutabagaResult<()> {
        let snapshot_save = kumquat_gpu_protocol_ctrl_hdr {
            type_: KUMQUAT_GPU_PROTOCOL_SNAPSHOT_SAVE,
            ..Default::default()
        };
        let mut stream = self.stream_lock.lock().unwrap();

        stream
            .write(KumquatGpuProtocolWrite::Cmd(snapshot_save))?;

        let mut protocols = stream.read()?;
        match protocols.remove(0) {
            KumquatGpuProtocol::RespOkSnapshot => Ok(()),
            _ => Err(RutabagaError::Unsupported),
        }
    }

    pub fn restore(&mut self) -> RutabagaResult<()> {
        let snapshot_restore = kumquat_gpu_protocol_ctrl_hdr {
            type_: KUMQUAT_GPU_PROTOCOL_SNAPSHOT_RESTORE,
            ..Default::default()
        };

        let mut stream = self.stream_lock.lock().unwrap();
        stream
            .write(KumquatGpuProtocolWrite::Cmd(snapshot_restore))?;

        let mut protocols = stream.read()?;
        match protocols.remove(0) {
            KumquatGpuProtocol::RespOkSnapshot => Ok(()),
            _ => Err(RutabagaError::Unsupported),
        }
    }
}

impl Drop for VirtGpuKumquat {
    fn drop(&mut self) {
        let mut stream = self.stream_lock.lock().unwrap();

        if self.context_id != 0 {
            for (_, resource) in self.resources.iter() {
                let detach_resource = kumquat_gpu_protocol_ctx_resource {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_CTX_DETACH_RESOURCE,
                        ..Default::default()
                    },
                    ctx_id: self.context_id,
                    resource_id: resource.resource_id,
                };

                let _ = stream
                    .write(KumquatGpuProtocolWrite::Cmd(detach_resource));
            }

            self.resources.clear();
            let context_destroy = kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_CTX_DESTROY,
                payload: self.context_id,
            };

            let _ = stream
                .write(KumquatGpuProtocolWrite::Cmd(context_destroy));
        }
    }
}
