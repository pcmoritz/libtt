use std::collections::BTreeSet;

const WORKER_Y_START: u8 = 2;
const WORKER_Y_END: u8 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
        let harvested = harvested_dram_banks
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut tiles = Vec::new();

        for bank in 0..Self::BANK_COUNT {
            if harvested.contains(&bank) {
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
    let mut cores = Vec::with_capacity(tensix_x.len() * (WORKER_Y_END - WORKER_Y_START) as usize);
    for &x in tensix_x {
        for y in WORKER_Y_START..WORKER_Y_END {
            cores.push(CoreCoord { x, y });
        }
    }
    cores
}
