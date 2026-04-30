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
    per_core: Vec<PerCoreRuntimeArgs>,
    current_core: Option<usize>,
    semaphores: usize,
}

impl RuntimeArgsBuilder {
    pub(crate) fn new(semaphores: usize) -> Self {
        Self {
            per_core: Vec::new(),
            current_core: None,
            semaphores,
        }
    }

    pub(crate) fn add_core(
        &mut self,
        core: CoreCoord,
        build: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.current_core.is_some() {
            return Err(invalid_input("runtime arg cores cannot be nested"));
        }

        let index = self.per_core.len();
        self.per_core.push(PerCoreRuntimeArgs::new(core));
        self.current_core = Some(index);
        let result = build(self);
        self.current_core = None;
        if result.is_err() {
            self.per_core.pop();
        }
        result
    }

    pub(crate) fn reader_arg(&mut self, value: u32) {
        self.current_core_mut().reader.push(value);
    }

    pub(crate) fn reader_dynamic_arg(&mut self) {
        self.current_core_mut().reader.push_dynamic();
    }

    pub(crate) fn writer_arg(&mut self, value: u32) {
        self.current_core_mut().writer.push(value);
    }

    pub(crate) fn writer_dynamic_arg(&mut self) {
        self.current_core_mut().writer.push_dynamic();
    }

    pub(crate) fn compute_arg(&mut self, value: u32) {
        self.current_core_mut().compute.push(value);
    }

    pub(crate) fn compute_dynamic_arg(&mut self) {
        self.current_core_mut().compute.push_dynamic();
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

    fn current_core_mut(&mut self) -> &mut PerCoreRuntimeArgs {
        let core_index = self
            .current_core
            .expect("runtime args must be added from RuntimeArgsBuilder::add_core");
        &mut self.per_core[core_index]
    }

    fn lower(mut self) -> io::Result<(RuntimeArgs, Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.per_core.sort_unstable_by_key(|args| args.core);

        let writer_args = self
            .per_core
            .first()
            .map(|args| args.writer.values.clone())
            .unwrap_or_default();
        let reader_args = self
            .per_core
            .first()
            .map(|args| args.reader.values.clone())
            .unwrap_or_default();
        let compute_args = self
            .per_core
            .first()
            .map(|args| args.compute.values.clone())
            .unwrap_or_default();

        let mut cores = Vec::with_capacity(self.per_core.len());
        let mut blobs = Vec::with_capacity(self.per_core.len());
        let mut writer_patches = Vec::with_capacity(self.per_core.len());
        let mut reader_patches = Vec::with_capacity(self.per_core.len());
        let mut compute_patches = Vec::with_capacity(self.per_core.len());

        for args in self.per_core {
            let writer_bytes = args.writer.len() * size_of::<u32>();
            let reader_bytes = args.reader.len() * size_of::<u32>();
            let compute_bytes = args.compute.len() * size_of::<u32>();
            let sem_off = align16(writer_bytes + reader_bytes + compute_bytes);

            writer_patches.push(section_patches(&args.writer, 0));
            reader_patches.push(section_patches(&args.reader, writer_bytes));
            compute_patches.push(section_patches(&args.compute, writer_bytes + reader_bytes));
            cores.push(args.core);
            blobs.push(pack_rta(
                &args.writer.values,
                &args.reader.values,
                &args.compute.values,
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

        Ok((runtime_args, writer_args, reader_args, compute_args))
    }
}

type RuntimeArgPatchGroup = (usize, Vec<Option<usize>>);

#[derive(Clone, Debug, PartialEq, Eq)]
struct PerCoreRuntimeArgs {
    core: CoreCoord,
    writer: RuntimeArgSection,
    reader: RuntimeArgSection,
    compute: RuntimeArgSection,
}

impl PerCoreRuntimeArgs {
    fn new(core: CoreCoord) -> Self {
        Self {
            core,
            writer: RuntimeArgSection::default(),
            reader: RuntimeArgSection::default(),
            compute: RuntimeArgSection::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RuntimeArgSection {
    values: Vec<u32>,
    dynamic_indices: Vec<usize>,
}

impl RuntimeArgSection {
    fn len(&self) -> usize {
        self.values.len()
    }

    // Static runtime args are fixed when the cached program is built.
    fn push(&mut self, value: u32) {
        self.values.push(value);
    }

    // Dynamic runtime args reserve a slot that is patched per launch.
    fn push_dynamic(&mut self) {
        let index = self.values.len();
        self.values.push(0);
        self.dynamic_indices.push(index);
    }
}

fn section_patches(args: &RuntimeArgSection, base_offset: usize) -> Vec<(usize, usize)> {
    args.dynamic_indices
        .iter()
        .map(|&index| (index, base_offset + index * size_of::<u32>()))
        .collect()
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
    groups.into_iter().collect()
}

fn patch_section(
    blobs: &mut [Vec<u8>],
    cores: &[CoreCoord],
    groups: &[RuntimeArgPatchGroup],
    value: impl Fn(CoreCoord, usize) -> Option<u32>,
) -> io::Result<()> {
    for group in groups {
        for (core_index, offset) in group.1.iter().enumerate() {
            let Some(offset) = offset else {
                continue;
            };
            let value = value(cores[core_index], group.0).ok_or_else(|| {
                invalid_input(format!(
                    "missing dynamic runtime arg value for index {} on core {}",
                    group.0, cores[core_index]
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
            .add_core(CoreCoord { x: 1, y: 2 }, |args| {
                args.writer_arg(7);
                args.writer_dynamic_arg();
                args.reader_dynamic_arg();
                args.reader_arg(9);
                Ok(())
            })
            .expect("add core");

        let mut runtime_args = builder.build().expect("lower");
        runtime_args
            .update_from_kernel(&TestKernel)
            .expect("update");

        assert_eq!(&runtime_args.blobs()[0][4..8], &0x1111u32.to_le_bytes());
        assert_eq!(&runtime_args.blobs()[0][8..12], &0x2222u32.to_le_bytes());
    }
}
