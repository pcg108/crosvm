// Copyright 2017 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#![cfg_attr(windows, allow(unused))]

//! Emulates virtual and hardware devices.

pub mod ac_adapter;
pub mod acpi;
pub mod bat;
mod bus;
#[cfg(feature = "stats")]
mod bus_stats;
pub mod cmos;
#[cfg(target_arch = "x86_64")]
mod debugcon;
mod fw_cfg;
mod i8042;
mod irq_event;
pub mod irqchip;
mod pci;
mod pflash;
pub mod pl030;
pub mod pmc_virt;
mod serial;
pub mod serial_device;
mod suspendable;
mod sys;
#[cfg(any(target_os = "android", target_os = "linux"))]
mod virtcpufreq;
#[cfg(any(target_os = "android", target_os = "linux"))]
mod virtcpufreq_v2;
pub mod virtio;
#[cfg(feature = "vtpm")]
mod vtpm_proxy;

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        mod pit;
        pub use self::pit::{Pit, PitError};
        pub mod tsc;
    }
}

use std::sync::Arc;

use anyhow::anyhow;
use anyhow::Context;
use base::debug;
use base::error;
use base::info;
use base::Tube;
use base::TubeError;
use cros_async::AsyncTube;
use cros_async::Executor;
use serde::Deserialize;
use serde::Serialize;
use vm_control::DeviceControlCommand;
use vm_control::DevicesState;
use vm_control::VmResponse;
use vm_memory::GuestMemory;

pub use self::acpi::ACPIPMFixedEvent;
pub use self::acpi::ACPIPMResource;
pub use self::bat::BatteryError;
pub use self::bat::GoldfishBattery;
pub use self::bus::Bus;
pub use self::bus::BusAccessInfo;
pub use self::bus::BusDevice;
pub use self::bus::BusDeviceObj;
pub use self::bus::BusDeviceSync;
pub use self::bus::BusRange;
pub use self::bus::BusResumeDevice;
pub use self::bus::BusType;
pub use self::bus::Error as BusError;
pub use self::bus::HotPlugBus;
pub use self::bus::HotPlugKey;
#[cfg(feature = "stats")]
pub use self::bus_stats::BusStatistics;
#[cfg(target_arch = "x86_64")]
pub use self::debugcon::Debugcon;
pub use self::fw_cfg::Error as FwCfgError;
pub use self::fw_cfg::FwCfgDevice;
pub use self::fw_cfg::FwCfgItemType;
pub use self::fw_cfg::FwCfgParameters;
pub use self::fw_cfg::FW_CFG_BASE_PORT;
pub use self::fw_cfg::FW_CFG_MAX_FILE_SLOTS;
pub use self::fw_cfg::FW_CFG_WIDTH;
pub use self::i8042::I8042Device;
pub use self::irq_event::IrqEdgeEvent;
pub use self::irq_event::IrqLevelEvent;
pub use self::irqchip::*;
pub use self::pci::BarRange;
pub use self::pci::CrosvmDeviceId;
pub use self::pci::GpeScope;
#[cfg(feature = "pci-hotplug")]
pub use self::pci::HotPluggable;
#[cfg(feature = "pci-hotplug")]
pub use self::pci::IntxParameter;
#[cfg(feature = "pci-hotplug")]
pub use self::pci::NetResourceCarrier;
pub use self::pci::PciAddress;
pub use self::pci::PciAddressError;
pub use self::pci::PciBarConfiguration;
pub use self::pci::PciBarIndex;
pub use self::pci::PciBus;
pub use self::pci::PciClassCode;
pub use self::pci::PciConfigIo;
pub use self::pci::PciConfigMmio;
pub use self::pci::PciDevice;
pub use self::pci::PciDeviceError;
pub use self::pci::PciInterruptPin;
pub use self::pci::PciMmioMapper;
pub use self::pci::PciRoot;
pub use self::pci::PciRootCommand;
pub use self::pci::PciVirtualConfigMmio;
pub use self::pci::PreferredIrq;
#[cfg(feature = "pci-hotplug")]
pub use self::pci::ResourceCarrier;
pub use self::pci::StubPciDevice;
pub use self::pci::StubPciParameters;
pub use self::pflash::Pflash;
pub use self::pflash::PflashParameters;
pub use self::pl030::Pl030;
pub use self::pmc_virt::VirtualPmc;
pub use self::serial::Serial;
pub use self::serial_device::Error as SerialError;
pub use self::serial_device::SerialDevice;
pub use self::serial_device::SerialHardware;
pub use self::serial_device::SerialParameters;
pub use self::serial_device::SerialType;
pub use self::suspendable::DeviceState;
pub use self::suspendable::Suspendable;
#[cfg(any(target_os = "android", target_os = "linux"))]
pub use self::virtcpufreq::VirtCpufreq;
#[cfg(any(target_os = "android", target_os = "linux"))]
pub use self::virtcpufreq_v2::VirtCpufreqV2;
pub use self::virtio::VirtioMmioDevice;
pub use self::virtio::VirtioPciDevice;
#[cfg(feature = "vtpm")]
pub use self::vtpm_proxy::VtpmProxy;

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "android", target_os = "linux"))] {
        mod platform;
        mod proxy;
        pub mod vmwdt;
        pub mod vfio;
        #[cfg(feature = "usb")]
        #[macro_use]
        mod register_space;
        #[cfg(feature = "usb")]
        pub mod usb;
        #[cfg(feature = "usb")]
        mod utils;

        pub use self::pci::{
            CoIommuDev, CoIommuParameters, CoIommuUnpinPolicy, PciBridge, PcieDownstreamPort,
            PcieHostPort, PcieRootPort, PcieUpstreamPort, PvPanicCode, PvPanicPciDevice,
            VfioPciDevice,
        };
        pub use self::platform::VfioPlatformDevice;
        pub use self::ac_adapter::AcAdapter;
        pub use self::proxy::ChildProcIntf;
        pub use self::proxy::Error as ProxyError;
        pub use self::proxy::ProxyDevice;
        #[cfg(feature = "usb")]
        pub use self::usb::backend::device_provider::DeviceProvider;
        #[cfg(feature = "usb")]
        pub use self::usb::xhci::xhci_controller::XhciController;
        pub use self::vfio::VfioContainer;
        pub use self::vfio::VfioDevice;
        pub use self::vfio::VfioDeviceType;
        pub use self::virtio::vfio_wrapper;

    } else if #[cfg(windows)] {
    } else {
        compile_error!("Unsupported platform");
    }
}

/// Request CoIOMMU to unpin a specific range.
#[derive(Serialize, Deserialize, Debug)]
pub struct UnpinRequest {
    /// The ranges presents (start gfn, count).
    ranges: Vec<(u64, u64)>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum UnpinResponse {
    Success,
    Failed,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
pub enum IommuDevType {
    #[serde(rename = "off")]
    #[default]
    NoIommu,
    #[serde(rename = "viommu")]
    VirtioIommu,
    #[serde(rename = "coiommu")]
    CoIommu,
    #[serde(rename = "pkvm-iommu")]
    PkvmPviommu,
}

// Thread that handles commands sent to devices - such as snapshot, sleep, suspend
// Created when the VM is first created, and re-created on resumption of the VM.
pub fn create_devices_worker_thread(
    guest_memory: GuestMemory,
    io_bus: Arc<Bus>,
    mmio_bus: Arc<Bus>,
    device_ctrl_resp: Tube,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("device_control".to_string())
        .spawn(move || {
            let ex = Executor::new().expect("Failed to create an executor");

            let async_control = AsyncTube::new(&ex, device_ctrl_resp).unwrap();
            match ex.run_until(async move {
                handle_command_tube(async_control, guest_memory, io_bus, mmio_bus).await
            }) {
                Ok(_) => {}
                Err(e) => {
                    error!("Device control thread exited with error: {}", e);
                }
            };
        })
}

fn sleep_buses(buses: &[&Bus]) -> anyhow::Result<()> {
    for bus in buses {
        bus.sleep_devices()
            .with_context(|| format!("failed to sleep devices on {:?} bus", bus.get_bus_type()))?;
        debug!("Devices slept successfully on {:?} bus", bus.get_bus_type());
    }
    Ok(())
}

fn wake_buses(buses: &[&Bus]) {
    for bus in buses {
        bus.wake_devices()
            .with_context(|| format!("failed to wake devices on {:?} bus", bus.get_bus_type()))
            // Some devices may have slept. Eternally.
            // Recovery - impossible.
            // Shut down VM.
            .expect("VM panicked to avoid unexpected behavior");
        debug!(
            "Devices awoken successfully on {:?} Bus",
            bus.get_bus_type()
        );
    }
}

// Use 64MB chunks when writing the memory snapshot (if encryption is used).
const MEMORY_SNAP_ENCRYPTED_CHUNK_SIZE_BYTES: usize = 1024 * 1024 * 64;

async fn snapshot_handler(
    snapshot_writer: vm_control::SnapshotWriter,
    guest_memory: &GuestMemory,
    buses: &[&Bus],
    compress_memory: bool,
) -> anyhow::Result<()> {
    // SAFETY:
    // VM & devices are stopped.
    let guest_memory_metadata = unsafe {
        guest_memory
            .snapshot(
                &mut snapshot_writer
                    .raw_fragment_with_chunk_size("mem", MEMORY_SNAP_ENCRYPTED_CHUNK_SIZE_BYTES)?,
                compress_memory,
            )
            .context("failed to snapshot memory")?
    };
    snapshot_writer.write_fragment("mem_metadata", &guest_memory_metadata)?;
    for (i, bus) in buses.iter().enumerate() {
        bus.snapshot_devices(&snapshot_writer.add_namespace(&format!("bus{i}"))?)
            .context("failed to snapshot bus devices")?;
        debug!(
            "Devices snapshot successfully for {:?} Bus",
            bus.get_bus_type()
        );
    }
    Ok(())
}

async fn restore_handler(
    snapshot_reader: vm_control::SnapshotReader,
    guest_memory: &GuestMemory,
    buses: &[&Bus],
) -> anyhow::Result<()> {
    let guest_memory_metadata = snapshot_reader.read_fragment("mem_metadata")?;
    // SAFETY:
    // VM & devices are stopped.
    unsafe {
        guest_memory.restore(
            guest_memory_metadata,
            &mut snapshot_reader.raw_fragment("mem")?,
        )?
    };
    for (i, bus) in buses.iter().enumerate() {
        bus.restore_devices(&snapshot_reader.namespace(&format!("bus{i}"))?)
            .context("failed to restore bus devices")?;
        debug!(
            "Devices restore successfully for {:?} Bus",
            bus.get_bus_type()
        );
    }
    Ok(())
}

async fn handle_command_tube(
    command_tube: AsyncTube,
    guest_memory: GuestMemory,
    io_bus: Arc<Bus>,
    mmio_bus: Arc<Bus>,
) -> anyhow::Result<()> {
    let buses = &[&*io_bus, &*mmio_bus];

    // We assume devices are awake. This is safe because if the VM starts the
    // sleeping state, run_control will ask us to sleep devices.
    let mut devices_state = DevicesState::Wake;

    loop {
        match command_tube.next().await {
            Ok(command) => {
                match command {
                    DeviceControlCommand::SleepDevices => {
                        if let DevicesState::Wake = devices_state {
                            match sleep_buses(buses) {
                                Ok(()) => {
                                    devices_state = DevicesState::Sleep;
                                }
                                Err(e) => {
                                    error!("failed to sleep: {:#}", e);

                                    // Failing to sleep could mean a single device failing to sleep.
                                    // Wake up devices to resume functionality of the VM.
                                    info!("Attempting to wake devices after failed sleep");
                                    wake_buses(buses);

                                    command_tube
                                        .send(VmResponse::ErrString(e.to_string()))
                                        .await
                                        .context("failed to send response.")?;
                                    continue;
                                }
                            }
                        }
                        command_tube
                            .send(VmResponse::Ok)
                            .await
                            .context("failed to reply to sleep command")?;
                    }
                    DeviceControlCommand::WakeDevices => {
                        if let DevicesState::Sleep = devices_state {
                            wake_buses(buses);
                            devices_state = DevicesState::Wake;
                        }
                        command_tube
                            .send(VmResponse::Ok)
                            .await
                            .context("failed to reply to wake devices request")?;
                    }
                    DeviceControlCommand::SnapshotDevices {
                        snapshot_writer,
                        compress_memory,
                    } => {
                        assert!(
                            matches!(devices_state, DevicesState::Sleep),
                            "devices must be sleeping to snapshot"
                        );
                        if let Err(e) =
                            snapshot_handler(snapshot_writer, &guest_memory, buses, compress_memory)
                                .await
                        {
                            error!("failed to snapshot: {:#}", e);
                            command_tube
                                .send(VmResponse::ErrString(e.to_string()))
                                .await
                                .context("Failed to send response")?;
                            continue;
                        }
                        command_tube
                            .send(VmResponse::Ok)
                            .await
                            .context("Failed to send response")?;
                    }
                    DeviceControlCommand::RestoreDevices { snapshot_reader } => {
                        assert!(
                            matches!(devices_state, DevicesState::Sleep),
                            "devices must be sleeping to restore"
                        );
                        if let Err(e) =
                            restore_handler(snapshot_reader, &guest_memory, &[&*io_bus, &*mmio_bus])
                                .await
                        {
                            error!("failed to restore: {:#}", e);
                            command_tube
                                .send(VmResponse::ErrString(e.to_string()))
                                .await
                                .context("Failed to send response")?;
                            continue;
                        }
                        command_tube
                            .send(VmResponse::Ok)
                            .await
                            .context("Failed to send response")?;
                    }
                    DeviceControlCommand::GetDevicesState => {
                        command_tube
                            .send(VmResponse::DevicesState(devices_state.clone()))
                            .await
                            .context("failed to send response")?;
                    }
                    DeviceControlCommand::Exit => {
                        return Ok(());
                    }
                };
            }
            Err(e) => {
                if matches!(e, TubeError::Disconnected) {
                    // Tube disconnected - shut down thread.
                    return Ok(());
                }
                return Err(anyhow!("Failed to receive: {}", e));
            }
        }
    }
}
