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
}

impl RuntimeArgsBuilder {
    pub(crate) fn new(semaphores: usize) -> Self {
        Self {
            per_core: BTreeMap::new(),
            semaphores,
        }
    }

    pub(crate) fn add_core(
        &mut self,
        core: CoreCoord,
        writer: Vec<Option<u32>>,
        reader: Vec<Option<u32>>,
        compute: Vec<Option<u32>>,
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
        let mut cores = Vec::with_capacity(self.per_core.len());
        let mut blobs = Vec::with_capacity(self.per_core.len());
        let mut writer_patches = Vec::with_capacity(self.per_core.len());
        let mut reader_patches = Vec::with_capacity(self.per_core.len());
        let mut compute_patches = Vec::with_capacity(self.per_core.len());
        let mut writer_args = None;
        let mut reader_args = None;
        let mut compute_args = None;

        for (core, args) in self.per_core {
            let writer_bytes = args.writer.len() * size_of::<u32>();
            let reader_bytes = args.reader.len() * size_of::<u32>();
            let compute_bytes = args.compute.len() * size_of::<u32>();
            let sem_off = align16(writer_bytes + reader_bytes + compute_bytes);

            let (writer, patches) = lower_section(&args.writer, 0);
            writer_patches.push(patches);
            writer_args.get_or_insert_with(|| writer.clone());

            let (reader, patches) = lower_section(&args.reader, writer_bytes);
            reader_patches.push(patches);
            reader_args.get_or_insert_with(|| reader.clone());

            let (compute, patches) = lower_section(&args.compute, writer_bytes + reader_bytes);
            compute_patches.push(patches);
            compute_args.get_or_insert_with(|| compute.clone());

            cores.push(core);
            blobs.push(pack_rta(
                &writer,
                &reader,
                &compute,
                self.semaphores,
                sem_off,
            ));
        }

        let runtime_args = RuntimeArgs {
            cores: cores.into(),
            writer_patches: patch_groups(writer_patches).into(),
            reader_patches: patch_groups(reader_patches).into(),
            compute_patches: patch_groups(compute_patches).into(),
            blobs,
        };

        Ok((
            runtime_args,
            writer_args.unwrap_or_default(),
            reader_args.unwrap_or_default(),
            compute_args.unwrap_or_default(),
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeArgPatchGroup {
    index: usize,
    offsets_by_core: Vec<Option<usize>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PerCoreRuntimeArgs {
    writer: Vec<Option<u32>>,
    reader: Vec<Option<u32>>,
    compute: Vec<Option<u32>>,
}

fn lower_section(args: &[Option<u32>], base_offset: usize) -> (Vec<u32>, Vec<(usize, usize)>) {
    let mut values = Vec::with_capacity(args.len());
    let mut patches = Vec::new();
    for (index, arg) in args.iter().enumerate() {
        match arg {
            Some(value) => values.push(*value),
            None => {
                values.push(0);
                patches.push((index, base_offset + index * size_of::<u32>()));
            }
        }
    }
    (values, patches)
}

fn patch_groups(per_core: Vec<Vec<(usize, usize)>>) -> Vec<RuntimeArgPatchGroup> {
    let core_count = per_core.len();
    let mut groups = BTreeMap::<usize, Vec<Option<usize>>>::new();
    for (core_index, patches) in per_core.into_iter().enumerate() {
        for (index, offset) in patches {
            groups
                .entry(index)
                .or_insert_with(|| vec![None; core_count])[core_index] = Some(offset);
        }
    }
    groups
        .into_iter()
        .map(|(index, offsets_by_core)| RuntimeArgPatchGroup {
            index,
            offsets_by_core,
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
        for (core_index, offset) in group.offsets_by_core.iter().enumerate() {
            let Some(offset) = offset else {
                continue;
            };
            let value = value(cores[core_index], group.index).ok_or_else(|| {
                invalid_input(format!(
                    "missing dynamic runtime arg value for index {} on core {}",
                    group.index, cores[core_index]
                ))
            })?;
            patch_u32(&mut blobs[core_index], *offset, value)?;
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
        let mut builder = RuntimeArgsBuilder::new(0);
        builder
            .add_core(
                CoreCoord { x: 1, y: 2 },
                vec![Some(7), None],
                vec![None, Some(9)],
                Vec::new(),
            )
            .expect("add core");

        let mut runtime_args = builder.build().expect("lower");
        runtime_args
            .update_from_kernel(&TestKernel)
            .expect("update");

        assert_eq!(&runtime_args.blobs()[0][4..8], &0x1111u32.to_le_bytes());
        assert_eq!(&runtime_args.blobs()[0][8..12], &0x2222u32.to_le_bytes());
    }
}
