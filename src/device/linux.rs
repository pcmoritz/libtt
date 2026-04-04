use super::{log, ARC_DEFAULT_TENSIX_ENABLED, CoreCoord, ProbeInfo};
use std::fs::{self, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const DEFAULT_GDDR_ENABLED: u32 = 0xff;
const TT_VENDOR: u32 = 0x1e52;
const BH_DEVICE: u32 = 0xb140;
const ARC_TILE: CoreCoord = CoreCoord { x: 8, y: 0 };
const ARC_NOC_BASE: u64 = 0x8000_0000;
const ARC_CSM_BASE: u64 = 0x1000_0000;
const ARC_CSM_SIZE: u64 = 1 << 19;
const SCRATCH_RAM_12: u64 = 0x30430;
const SCRATCH_RAM_13: u64 = 0x30434;
const TAG_TENSIX_ENABLED_COL: u16 = 34;
const TAG_GDDR_ENABLED: u16 = 36;
const PCI_COMMAND: u64 = 0x04;
const PCI_COMMAND_MEMORY: u16 = 0x02;
const PCI_COMMAND_MASTER: u16 = 0x04;
const TLB_2M_SIZE: u64 = 1 << 21;
const TLB_REGS_START: u64 = 0x1fc0_0000;
const TLB_REG_SIZE: u64 = 12;
const TLB_STRIDE_OFFSET: u64 = 210 * TLB_REG_SIZE;

pub(super) fn detect_probe_info(index: usize) -> Option<ProbeInfo> {
    match probe_info_for_index(index) {
        Ok(probe) => {
            if probe.is_none() {
                log(format!("linux probe index={index} no matching blackhole PCI device"));
            }
            probe
        }
        Err(err) => {
            log(format!("linux probe index={index} failed: {err}"));
            None
        }
    }
}

fn probe_info_for_index(index: usize) -> io::Result<Option<ProbeInfo>> {
    let sysfs_paths = blackhole_sysfs_paths()?;
    log(format!(
        "linux probe available_blackhole_pci_devices={}",
        sysfs_paths.len()
    ));
    let Some(sysfs_path) = sysfs_paths.into_iter().nth(index) else {
        return Ok(None);
    };

    log(format!(
        "linux probe index={index} sysfs_path={}",
        sysfs_path.display()
    ));
    let probe = ProbeDevice::open(&sysfs_path)?;
    let (gddr_enabled_mask, tensix_enabled_col_mask) = probe.read_arc_enabled_masks()?;
    log(format!(
        "linux probe index={index} tensix_enabled_col_mask=0x{tensix_enabled_col_mask:08x} gddr_enabled_mask=0x{gddr_enabled_mask:08x}"
    ));
    Ok(Some(ProbeInfo {
        tensix_enabled_col_mask,
        gddr_enabled_mask,
    }))
}

fn blackhole_sysfs_paths() -> io::Result<Vec<PathBuf>> {
    let mut devices = Vec::new();

    for entry in fs::read_dir("/sys/bus/pci/devices")? {
        let entry = entry?;
        let path = entry.path();
        let vendor = read_hex_u32(&path.join("vendor"))?;
        let device = read_hex_u32(&path.join("device"))?;
        if vendor == TT_VENDOR && device == BH_DEVICE {
            devices.push(path);
        }
    }

    devices.sort();
    Ok(devices)
}

fn read_hex_u32(path: &Path) -> io::Result<u32> {
    let text = fs::read_to_string(path)?;
    let trimmed = text.trim();
    let hex = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    u32::from_str_radix(hex, 16).map_err(io::Error::other)
}

struct ProbeDevice {
    config: std::fs::File,
    bar0: std::fs::File,
}

impl ProbeDevice {
    fn open(sysfs_path: &Path) -> io::Result<Self> {
        ensure_pci_device_enabled(sysfs_path)?;

        let config = OpenOptions::new()
            .read(true)
            .write(true)
            .open(sysfs_path.join("config"))
            .map_err(|err| io::Error::new(err.kind(), format!("open config: {err}")))?;
        let bar0 = OpenOptions::new()
            .read(true)
            .write(true)
            .open(sysfs_path.join("resource0"))
            .map_err(|err| io::Error::new(err.kind(), format!("open resource0: {err}")))?;

        let probe = Self { config, bar0 };
        probe.enable_memory_and_bus_mastering()?;
        log(format!(
            "linux probe opened config/bar0 for {}",
            sysfs_path.display()
        ));
        Ok(probe)
    }

    fn enable_memory_and_bus_mastering(&self) -> io::Result<()> {
        let mut cmd = [0u8; 2];
        read_exact_at(&self.config, &mut cmd, PCI_COMMAND)
            .map_err(|err| io::Error::new(err.kind(), format!("read PCI_COMMAND: {err}")))?;
        let value = u16::from_le_bytes(cmd) | PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER;
        write_all_at(&self.config, &value.to_le_bytes(), PCI_COMMAND)
            .map_err(|err| io::Error::new(err.kind(), format!("write PCI_COMMAND: {err}")))?;
        Ok(())
    }

    fn read_arc_enabled_masks(&self) -> io::Result<(u32, u32)> {
        let table_base = self.read_arc_apb32(SCRATCH_RAM_13)? as u64;
        let data_base = self.read_arc_apb32(SCRATCH_RAM_12)? as u64;
        log(format!(
            "linux probe telemetry pointers table=0x{table_base:x} data=0x{data_base:x}"
        ));

        if !is_arc_csm_addr(table_base, 4) || !is_arc_csm_addr(data_base, 4) {
            return Err(io::Error::other(format!(
                "ARC not ready: telemetry pointers table=0x{table_base:x} data=0x{data_base:x}"
            )));
        }

        let entry_count = self.read_arc_noc32(table_base + 4)? as usize;
        log(format!("linux probe telemetry entry_count={entry_count}"));
        if entry_count == 0 || entry_count > 4096 {
            return Err(io::Error::other(format!(
                "invalid ARC telemetry entry_count 0x{entry_count:x} at 0x{table_base:x}"
            )));
        }

        let mut tensix_enabled = ARC_DEFAULT_TENSIX_ENABLED;
        let mut gddr_enabled = DEFAULT_GDDR_ENABLED;

        for i in 0..entry_count {
            let tag_offset = self.read_arc_noc32(table_base + 8 + (i as u64) * 4)?;
            let tag = (tag_offset & 0xffff) as u16;
            let offset_words = (tag_offset >> 16) & 0xffff;
            let value_addr = data_base + (offset_words as u64) * 4;

            if tag == TAG_TENSIX_ENABLED_COL {
                tensix_enabled = self.read_arc_noc32(value_addr)?;
            } else if tag == TAG_GDDR_ENABLED {
                gddr_enabled = self.read_arc_noc32(value_addr)?;
            }
        }

        Ok((gddr_enabled, tensix_enabled))
    }

    fn read_arc_apb32(&self, offset: u64) -> io::Result<u32> {
        self.configure_tlb(0, ARC_NOC_BASE, ARC_TILE, ARC_TILE)?;
        self.read_bar0_u32(offset)
    }

    fn read_arc_noc32(&self, addr: u64) -> io::Result<u32> {
        let base = addr & !(TLB_2M_SIZE - 1);
        let offset = addr - base;
        self.configure_tlb(0, base, ARC_TILE, ARC_TILE)?;
        self.read_bar0_u32(offset)
    }

    fn read_bar0_u32(&self, offset: u64) -> io::Result<u32> {
        let mut bytes = [0u8; 4];
        read_exact_at(&self.bar0, &mut bytes, offset)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn configure_tlb(&self, index: u64, addr: u64, start: CoreCoord, end: CoreCoord) -> io::Result<()> {
        let reg_offset = TLB_REGS_START + index * TLB_REG_SIZE;
        let local_offset = addr >> 21;
        let value = (local_offset as u128)
            | ((end.x as u128) << 43)
            | ((end.y as u128) << 49)
            | ((start.x as u128) << 55)
            | ((start.y as u128) << 61)
            | (1u128 << 70);
        let lo = (value & 0xffff_ffff) as u32;
        let mid = ((value >> 32) & 0xffff_ffff) as u32;
        let hi = ((value >> 64) & 0xffff_ffff) as u32;

        write_all_at(&self.bar0, &lo.to_le_bytes(), reg_offset)?;
        write_all_at(&self.bar0, &mid.to_le_bytes(), reg_offset + 4)?;
        write_all_at(&self.bar0, &hi.to_le_bytes(), reg_offset + 8)?;

        if index < 32 {
            write_all_at(
                &self.bar0,
                &0u32.to_le_bytes(),
                TLB_REGS_START + TLB_STRIDE_OFFSET + index * 4,
            )?;
        }

        Ok(())
    }
}

fn ensure_pci_device_enabled(sysfs_path: &Path) -> io::Result<()> {
    let enable_path = sysfs_path.join("enable");
    let current = fs::read_to_string(&enable_path)
        .map_err(|err| io::Error::new(err.kind(), format!("read enable: {err}")))?;

    if current.trim() == "1" {
        log(format!(
            "linux probe pci device already enabled: {}",
            enable_path.display()
        ));
        return Ok(());
    }

    log(format!(
        "linux probe enabling pci device via {}",
        enable_path.display()
    ));
    fs::write(&enable_path, "1")
        .map_err(|err| io::Error::new(err.kind(), format!("write enable: {err}")))?;
    Ok(())
}

fn is_arc_csm_addr(addr: u64, length: u64) -> bool {
    addr >= ARC_CSM_BASE && addr + length <= ARC_CSM_BASE + ARC_CSM_SIZE
}

fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let read = file.read_at(buf, offset)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short read from sysfs resource",
            ));
        }
        offset += read as u64;
        buf = &mut buf[read..];
    }
    Ok(())
}

fn write_all_at(file: &std::fs::File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let written = file.write_at(buf, offset)?;
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short write to sysfs resource",
            ));
        }
        offset += written as u64;
        buf = &buf[written..];
    }
    Ok(())
}
