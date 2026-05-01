use crate::dispatch::{pack_rta, Program};
use crate::hw::CoreCoord;
use std::collections::BTreeMap;
use std::io;
use std::mem::size_of;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeArgs {
    cores: Arc<[CoreCoord]>,
    writer_patches: Arc<[RuntimeArgPatchGroup]>,
    reader_patches: Arc<[RuntimeArgPatchGroup]>,
    compute_patches: Arc<[RuntimeArgPatchGroup]>,
    blobs: Vec<Vec<u8>>,
}

impl RuntimeArgs {
    pub(crate) fn cores(&self) -> &[CoreCoord] {
        &self.cores
    }

    pub(crate) fn blobs(&self) -> &[Vec<u8>] {
        &self.blobs
    }

    #[inline]
    pub(crate) fn update_from_kernel(&mut self, kernel: &impl Kernel) -> io::Result<()> {
        patch_section(
            &mut self.blobs,
            &self.cores,
            &self.reader_patches,
            |core, index| kernel.reader_runtime_arg(core, index),
        )?;
        patch_section(
            &mut self.blobs,
            &self.cores,
            &self.writer_patches,
            |core, index| kernel.writer_runtime_arg(core, index),
        )?;
        patch_section(
            &mut self.blobs,
            &self.cores,
            &self.compute_patches,
            |core, index| kernel.compute_runtime_arg(core, index),
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeArgsBuilder {
    per_core: BTreeMap<CoreCoord, PerCoreRuntimeArgs>,
    semaphores: usize,
    writer_dynamic_indices: Vec<usize>,
    reader_dynamic_indices: Vec<usize>,
    compute_dynamic_indices: Vec<usize>,
}

impl RuntimeArgsBuilder {
    pub(crate) fn new(
        semaphores: usize,
        writer_dynamic_indices: Vec<usize>,
        reader_dynamic_indices: Vec<usize>,
        compute_dynamic_indices: Vec<usize>,
    ) -> Self {
        Self {
            per_core: BTreeMap::new(),
            semaphores,
            writer_dynamic_indices,
            reader_dynamic_indices,
            compute_dynamic_indices,
        }
    }

    pub(crate) fn add_core(
        &mut self,
        core: CoreCoord,
        writer: Vec<u32>,
        reader: Vec<u32>,
        compute: Vec<u32>,
    ) -> io::Result<()> {
        if self.per_core.contains_key(&core) {
            return Err(invalid_input(format!(
                "duplicate runtime args for core {core}"
            )));
        }

        self.per_core.insert(
            core,
            PerCoreRuntimeArgs {
                writer,
                reader,
                compute,
            },
        );
        Ok(())
    }

    pub(crate) fn build(self) -> io::Result<RuntimeArgs> {
        Ok(self.lower()?.0)
    }

    pub(crate) fn apply_to_program(self, program: &mut Program) -> io::Result<()> {
        let semaphores = self.semaphores;
        let (runtime_args, writer_args, reader_args, compute_args) = self.lower()?;
        program.writer_args = writer_args;
        program.reader_args = reader_args;
        program.compute_args = compute_args;
        program.semaphores = semaphores;
        program.runtime_args = Some(runtime_args);
        Ok(())
    }

    fn lower(self) -> io::Result<(RuntimeArgs, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let Some(layout) = self.per_core.values().next().cloned() else {
            return Err(invalid_input("runtime args require at least one core"));
        };
        let writer_args = layout.writer.clone();
        let reader_args = layout.reader.clone();
        let compute_args = layout.compute.clone();
        let writer_bytes = layout.writer.len() * size_of::<u32>();
        let reader_bytes = layout.reader.len() * size_of::<u32>();
        let compute_bytes = layout.compute.len() * size_of::<u32>();
        let sem_off = align16(writer_bytes + reader_bytes + compute_bytes);

        let writer_patches = section_patches(&self.writer_dynamic_indices, 0);
        let reader_patches = section_patches(&self.reader_dynamic_indices, writer_bytes);
        let compute_patches =
            section_patches(&self.compute_dynamic_indices, writer_bytes + reader_bytes);

        let mut cores = Vec::with_capacity(self.per_core.len());
        let mut blobs = Vec::with_capacity(self.per_core.len());

        for (core, args) in self.per_core {
            cores.push(core);
            blobs.push(pack_rta(
                &args.writer,
                &args.reader,
                &args.compute,
                self.semaphores,
                sem_off,
            ));
        }

        let runtime_args = RuntimeArgs {
            cores: cores.into(),
            writer_patches: writer_patches.into(),
            reader_patches: reader_patches.into(),
            compute_patches: compute_patches.into(),
            blobs,
        };

        Ok((runtime_args, writer_args, reader_args, compute_args))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeArgPatchGroup {
    index: usize,
    offset: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PerCoreRuntimeArgs {
    writer: Vec<u32>,
    reader: Vec<u32>,
    compute: Vec<u32>,
}

fn section_patches(indices: &[usize], base_offset: usize) -> Vec<RuntimeArgPatchGroup> {
    indices
        .iter()
        .map(|&index| RuntimeArgPatchGroup {
            index,
            offset: base_offset + index * size_of::<u32>(),
        })
        .collect()
}

fn patch_section(
    blobs: &mut [Vec<u8>],
    cores: &[CoreCoord],
    groups: &[RuntimeArgPatchGroup],
    value: impl Fn(CoreCoord, usize) -> Option<u32>,
) -> io::Result<()> {
    for group in groups {
        for (core_index, blob) in blobs.iter_mut().enumerate() {
            let value = value(cores[core_index], group.index).ok_or_else(|| {
                invalid_input(format!(
                    "missing dynamic runtime arg value for index {} on core {}",
                    group.index, cores[core_index]
                ))
            })?;
            patch_u32(blob, group.offset, value)?;
        }
    }
    Ok(())
}

#[inline]
fn patch_u32(blob: &mut [u8], offset: usize, value: u32) -> io::Result<()> {
    let end = offset + size_of::<u32>();
    let blob_len = blob.len();
    let bytes = blob.get_mut(offset..end).ok_or_else(|| {
        invalid_input(format!(
            "runtime arg patch offset {offset} exceeds blob size {blob_len}"
        ))
    })?;
    bytes.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn align16(value: usize) -> usize {
    (value + 15) & !15
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

pub(crate) trait Kernel {
    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, _index: usize) -> Option<u32> {
        None
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, _index: usize) -> Option<u32> {
        None
    }

    #[inline]
    fn compute_runtime_arg(&self, _core: CoreCoord, _index: usize) -> Option<u32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestKernel;

    impl Kernel for TestKernel {
        fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
            (index == 0).then_some(0x2222)
        }

        fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
            (index == 1).then_some(0x1111)
        }
    }

    #[test]
    fn runtime_args_update_patches_blobs_by_section_and_index() {
        let mut builder = RuntimeArgsBuilder::new(0, vec![1], vec![0], Vec::new());
        builder
            .add_core(CoreCoord { x: 1, y: 2 }, vec![7, 0], vec![0, 9], Vec::new())
            .expect("add core");

        let mut runtime_args = builder.build().expect("lower");
        runtime_args
            .update_from_kernel(&TestKernel)
            .expect("update");

        assert_eq!(&runtime_args.blobs()[0][4..8], &0x1111u32.to_le_bytes());
        assert_eq!(&runtime_args.blobs()[0][8..12], &0x2222u32.to_le_bytes());
    }
}
