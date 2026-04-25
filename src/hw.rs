#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CoreCoord {
    pub(crate) x: u8,
    pub(crate) y: u8,
}

impl std::fmt::Display for CoreCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.x, self.y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DramTile {
    pub(crate) bank: usize,
    pub(crate) x: u8,
    pub(crate) y: u8,
}

pub(crate) struct Arc;

impl Arc {
    pub(crate) const TILE: CoreCoord = CoreCoord { x: 8, y: 0 };
    pub(crate) const NOC_BASE: u64 = 0x8000_0000;
    pub(crate) const SCRATCH_RAM_13: usize = 0x30434;
    pub(crate) const DEFAULT_TENSIX_ENABLED: u32 = 0x3fff;
    pub(crate) const TAG_TENSIX_ENABLED_COL: u16 = 34;
    pub(crate) const TAG_GDDR_ENABLED: u16 = 36;
    pub(crate) const DEFAULT_GDDR_ENABLED: u32 = 0xff;
    pub(crate) const TLB_SIZE_2M: u64 = 1 << 21;

    pub(crate) fn active_tensix_core_count(enabled_col_mask: u32) -> usize {
        (enabled_col_mask & Self::DEFAULT_TENSIX_ENABLED).count_ones() as usize * 10
    }
}

pub(crate) struct TensixMMIO;

impl TensixMMIO {
    pub(crate) const LOCAL_RAM_START: u32 = 0xFFB00000;
    pub(crate) const LOCAL_RAM_END: u32 = 0xFFB01FFF;
    pub(crate) const RISCV_DEBUG_REG_SOFT_RESET_0: u64 = 0xFFB121B0;
    pub(crate) const RISCV_DEBUG_REG_TRISC0_RESET_PC: u64 = 0xFFB12228;
    pub(crate) const RISCV_DEBUG_REG_TRISC1_RESET_PC: u64 = 0xFFB1222C;
    pub(crate) const RISCV_DEBUG_REG_TRISC2_RESET_PC: u64 = 0xFFB12230;
    pub(crate) const RISCV_DEBUG_REG_NCRISC_RESET_PC: u64 = 0xFFB12238;
    pub(crate) const SOFT_RESET_ALL: u32 = 0x47800;
    pub(crate) const SOFT_RESET_BRISC_ONLY_RUN: u32 = 0x47000;
}

pub(crate) struct TensixL1;

impl TensixL1 {
    pub(crate) const SIZE: u32 = 0x180000;
    pub(crate) const LAUNCH: u32 = 0x000070; // mailbox_base(0x60) + 0x10
    pub(crate) const GO_MSG: u32 = 0x000370; // mailbox_base + 0x310
    pub(crate) const GO_MSG_INDEX: u32 = 0x0003A0; // mailbox_base + 0x340
    pub(crate) const KERNEL_CONFIG_BASE: u32 = 0x0086B0;
    pub(crate) const BRISC_FIRMWARE_BASE: u32 = 0x003840;
    pub(crate) const DATA_BUFFER_SPACE_BASE: u32 = 0x037000;
    pub(crate) const PROFILER_HOST_BUFFER_BYTES_PER_RISC: u32 = 65536;
    pub(crate) const MEM_BANK_TO_NOC_SCRATCH: u32 = 0x0116B0;
}

pub(crate) struct Dram;

impl Dram {
    pub(crate) const BANK_COUNT: usize = 8;
    pub(crate) const TILES_PER_BANK: usize = 3;
    pub(crate) const WRITE_OFFSET: u64 = 0x40;
    pub(crate) const BARRIER_BASE: usize = 0;
    pub(crate) const ALIGNMENT: usize = 64;
    pub(crate) const BARRIER_FLAGS: [u32; 2] = [0xaa, 0xbb];
    pub(crate) const TLB_SIZE_4G: u64 = 1 << 32;
    pub(crate) const BANK_TILE_YS: [[u8; 3]; Self::BANK_COUNT] = [
        [0, 1, 11],
        [2, 3, 10],
        [4, 8, 9],
        [5, 6, 7],
        [0, 1, 11],
        [2, 3, 10],
        [4, 8, 9],
        [5, 6, 7],
    ];

    pub(crate) fn active_banks(gddr_enabled_mask: u32) -> usize {
        (0..Self::BANK_COUNT)
            .filter(|bank| ((gddr_enabled_mask >> bank) & 1) != 0)
            .count()
    }

    pub(crate) fn harvested_banks(gddr_enabled_mask: u32) -> Vec<usize> {
        (0..Self::BANK_COUNT)
            .filter(|bank| ((gddr_enabled_mask >> bank) & 1) == 0)
            .collect()
    }

    pub(crate) fn tiles(harvested_dram_banks: &[usize]) -> Vec<DramTile> {
        let mut tiles = Vec::new();

        for bank in 0..Self::BANK_COUNT {
            if harvested_dram_banks.contains(&bank) {
                continue;
            }

            let x = Self::bank_x(bank);
            for &y in &Self::BANK_TILE_YS[bank] {
                tiles.push(DramTile { bank, x, y });
            }
        }

        tiles
    }

    fn bank_x(bank: usize) -> u8 {
        if bank < 4 { 0 } else { 9 }
    }
}

pub(crate) fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

pub(crate) fn align_down(value: u64, alignment: u64) -> (u64, u64) {
    let base = value & !(alignment - 1);
    (base, value - base)
}

pub(crate) fn worker_cores(tensix_x: &[u8]) -> Vec<CoreCoord> {
    let mut cores = Vec::with_capacity(tensix_x.len() * 10);
    for &x in tensix_x {
        for y in 2..12 {
            cores.push(CoreCoord { x, y });
        }
    }
    cores
}

pub(crate) fn noc_xy(x: u8, y: u8) -> u32 {
    (((y as u32) << 6) | x as u32) & 0xffff
}
