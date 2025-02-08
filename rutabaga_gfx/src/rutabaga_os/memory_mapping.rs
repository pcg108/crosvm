// Copyright 2023 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::rutabaga_os::sys::platform::MemoryMapping as PlatformMapping;
use crate::rutabaga_os::OwnedDescriptor;
use crate::rutabaga_utils::RutabagaMapping;
use crate::rutabaga_utils::RutabagaResult;

pub struct MemoryMapping {
    pub mapping: PlatformMapping,
}

impl MemoryMapping {
    pub fn from_safe_descriptor(
        descriptor: OwnedDescriptor,
        size: usize,
        map_info: u32,
    ) -> RutabagaResult<MemoryMapping> {
        let mapping = PlatformMapping::from_safe_descriptor(descriptor, size, map_info)?;
        Ok(MemoryMapping { mapping })
    }

    pub fn as_rutabaga_mapping(&self) -> RutabagaMapping {
        RutabagaMapping {
            ptr: self.mapping.addr.as_ptr() as u64,
            size: self.mapping.size as u64,
        }
    }
}
