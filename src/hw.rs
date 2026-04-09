use std::collections::BTreeSet;
use std::io;

const WORKER_Y_START: u8 = 2;
const WORKER_Y_END: u8 = 12;
const BANK_NOCS: usize = 2;
const BANK_PORT_STRIDE: u8 = 3;
const DRAM_NOC_LEFT_X: u8 = 17;
const DRAM_NOC_RIGHT_X: u8 = 18;

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

    pub(crate) fn build_bank_noc_table(
        harvested_dram_banks: &[usize],
        worker_cores: &[CoreCoord],
    ) -> io::Result<Vec<u8>> {
        let bank_ports = [[2, 1], [0, 1], [0, 1], [0, 1], [2, 1], [2, 1], [2, 1], [2, 1]];
        let num_dram_banks = Self::BANK_COUNT
            .checked_sub(harvested_dram_banks.len())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "too many harvested DRAM banks")
            })?;
        let bank_xy = Self::logical_bank_xy(harvested_dram_banks)?;

        let mut out = Vec::with_capacity(
            (BANK_NOCS * (num_dram_banks + worker_cores.len())) * 2
                + (num_dram_banks + worker_cores.len()) * 4,
        );

        for noc in 0..BANK_NOCS {
            for bank in 0..num_dram_banks {
                let (x, y0) = bank_xy[bank];
                out.extend_from_slice(&noc_xy(x, y0 + bank_ports[bank][noc]).to_le_bytes());
            }
        }

        let cols = worker_cores
            .iter()
            .map(|core| core.x)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if cols.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker core list must not be empty",
            ));
        }

        for _ in 0..BANK_NOCS {
            for index in 0..worker_cores.len() {
                let x = cols[index % cols.len()];
                let y = WORKER_Y_START
                    + ((index / cols.len()) % (WORKER_Y_END - WORKER_Y_START) as usize) as u8;
                out.extend_from_slice(&noc_xy(x, y).to_le_bytes());
            }
        }

        for _ in 0..(num_dram_banks + worker_cores.len()) {
            out.extend_from_slice(&0i32.to_le_bytes());
        }

        Ok(out)
    }

    fn logical_bank_xy(harvested_dram_banks: &[usize]) -> io::Result<Vec<(u8, u8)>> {
        match harvested_dram_banks {
            [] => Ok((0..Self::BANK_COUNT)
                .map(|bank| {
                    let x = if bank < Self::BANK_COUNT / 2 {
                        DRAM_NOC_LEFT_X
                    } else {
                        DRAM_NOC_RIGHT_X
                    };
                    let y = 12 + (bank % (Self::BANK_COUNT / 2)) as u8 * BANK_PORT_STRIDE;
                    (x, y)
                })
                .collect()),
            [harvested_bank] => {
                let half = Self::BANK_COUNT / 2;
                let mirror = if *harvested_bank < half {
                    harvested_bank + half - 1
                } else {
                    harvested_bank - half
                };

                let right = if *harvested_bank < half {
                    (0..(half - 1)).collect::<Vec<_>>()
                } else {
                    (half..(Self::BANK_COUNT - 1)).collect::<Vec<_>>()
                };
                let left = if *harvested_bank < half {
                    ((half - 1)..(Self::BANK_COUNT - 1))
                        .filter(|bank| *bank != mirror)
                        .chain(std::iter::once(mirror))
                        .collect::<Vec<_>>()
                } else {
                    (0..half)
                        .filter(|bank| *bank != mirror)
                        .chain(std::iter::once(mirror))
                        .collect::<Vec<_>>()
                };

                let mut bank_xy = vec![(0, 0); Self::BANK_COUNT - 1];
                for (index, bank) in right.into_iter().enumerate() {
                    bank_xy[bank] = (DRAM_NOC_RIGHT_X, 12 + index as u8 * BANK_PORT_STRIDE);
                }
                for (index, bank) in left.into_iter().enumerate() {
                    bank_xy[bank] = (DRAM_NOC_LEFT_X, 12 + index as u8 * BANK_PORT_STRIDE);
                }
                Ok(bank_xy)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported harvested DRAM bank count: {}",
                    harvested_dram_banks.len()
                ),
            )),
        }
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

pub(crate) fn noc_xy(x: u8, y: u8) -> u16 {
    (((y as u16) << 6) | x as u16) & 0xffff
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

#[cfg(test)]
mod tests {
    use super::*;

    const P100_TENSIX_X: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14];
    const P150_TENSIX_X: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16];

    fn decode_u16s(bytes: &[u8], count: usize) -> Vec<u16> {
        bytes[..count * 2]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunk length is fixed")))
            .collect()
    }

    #[test]
    fn build_bank_noc_table_matches_p100_layout() {
        let workers = worker_cores(&P100_TENSIX_X);
        let table = Dram::build_bank_noc_table(&[7], &workers).expect("table should build");
        let entries = decode_u16s(&table, (7 + workers.len()) * BANK_NOCS);

        assert_eq!(table.len(), (7 + workers.len()) * 8);
        assert_eq!(&entries[..7], &[913, 977, 1169, 1361, 914, 1106, 1298]);
        assert_eq!(&entries[7..14], &[849, 1041, 1233, 1425, 850, 1042, 1234]);
        assert_eq!(&entries[14..20], &[129, 130, 131, 132, 133, 134]);
    }

    #[test]
    fn build_bank_noc_table_matches_p150_layout() {
        let workers = worker_cores(&P150_TENSIX_X);
        let table = Dram::build_bank_noc_table(&[], &workers).expect("table should build");
        let entries = decode_u16s(&table, (8 + workers.len()) * BANK_NOCS);

        assert_eq!(table.len(), (8 + workers.len()) * 8);
        assert_eq!(
            &entries[..8],
            &[913, 977, 1169, 1361, 914, 1106, 1298, 1490]
        );
        assert_eq!(
            &entries[8..16],
            &[849, 1041, 1233, 1425, 850, 1042, 1234, 1426]
        );
        assert_eq!(&entries[16..22], &[129, 130, 131, 132, 133, 134]);
    }

    #[test]
    fn build_bank_noc_table_rejects_multiple_harvested_banks() {
        let workers = worker_cores(&P100_TENSIX_X);
        let err = Dram::build_bank_noc_table(&[0, 7], &workers).expect_err("layout should fail");
        assert!(
            err.to_string()
                .contains("unsupported harvested DRAM bank count")
        );
    }
}
