// Copyright 2018 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! A crate for handling 2D and 3D virtio-gpu hypercalls, along with graphics
//! swapchain allocation and mapping.

mod cross_domain;
mod generated;
mod gfxstream;
mod gfxstream_stub;
#[macro_use]
mod macros;
mod bytestream;
mod ipc;
#[cfg(any(feature = "gfxstream", feature = "virgl_renderer"))]
mod renderer_utils;
mod rutabaga_2d;
mod rutabaga_core;
mod rutabaga_gralloc;
mod rutabaga_os;
mod rutabaga_snapshot;
mod rutabaga_utils;
mod virgl_renderer;

pub use crate::rutabaga_core::calculate_capset_mask;
pub use crate::rutabaga_core::calculate_capset_names;
pub use crate::rutabaga_core::Rutabaga;
pub use crate::rutabaga_core::RutabagaBuilder;
pub use crate::rutabaga_gralloc::DrmFormat;
pub use crate::rutabaga_gralloc::ImageAllocationInfo;
pub use crate::rutabaga_gralloc::ImageMemoryRequirements;
pub use crate::rutabaga_gralloc::RutabagaGralloc;
pub use crate::rutabaga_gralloc::RutabagaGrallocBackendFlags;
pub use crate::rutabaga_gralloc::RutabagaGrallocFlags;
pub use crate::rutabaga_os::AsRawDescriptor;
pub use crate::rutabaga_os::FromRawDescriptor as RutabagaFromRawDescriptor;
pub use crate::rutabaga_os::IntoRawDescriptor as RutabagaIntoRawDescriptor;
pub use crate::rutabaga_os::MappedRegion as RutabagaMappedRegion;
pub use crate::rutabaga_os::OwnedDescriptor as RutabagaDescriptor;
pub use crate::rutabaga_os::RawDescriptor as RutabagaRawDescriptor;
pub use crate::rutabaga_utils::*;

pub mod kumquat_support {
    pub use crate::bytestream::Reader as RutabagaReader;
    pub use crate::bytestream::Writer as RutabagaWriter;
    pub use crate::ipc::kumquat_gpu_protocol;
    pub use crate::ipc::RutabagaStream;
    pub use crate::rutabaga_os::Event as RutabagaEvent;
    pub use crate::rutabaga_os::Listener as RutabagaListener;
    pub use crate::rutabaga_os::MemoryMapping as RutabagaMemoryMapping;
    pub use crate::rutabaga_os::SharedMemory as RutabagaSharedMemory;
    pub use crate::rutabaga_os::Tube as RutabagaTube;
    pub use crate::rutabaga_os::TubeType as RutabagaTubeType;
    pub use crate::rutabaga_os::WaitContext as RutabagaWaitContext;
}
