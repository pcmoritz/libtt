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
        let Some(layout) = self.per_core.values().next().cloned() else {
            return Ok((RuntimeArgs::empty(), Vec::new(), Vec::new(), Vec::new()));
        };
        let writer_args = lower_values(&layout.writer);
        let reader_args = lower_values(&layout.reader);
        let compute_args = lower_values(&layout.compute);
        let writer_bytes = layout.writer.len() * size_of::<u32>();
        let reader_bytes = layout.reader.len() * size_of::<u32>();
        let compute_bytes = layout.compute.len() * size_of::<u32>();
        let sem_off = align16(writer_bytes + reader_bytes + compute_bytes);

        let writer_patches = section_patches(&layout.writer, 0);
        let reader_patches = section_patches(&layout.reader, writer_bytes);
        let compute_patches = section_patches(&layout.compute, writer_bytes + reader_bytes);

        let mut cores = Vec::with_capacity(self.per_core.len());
        let mut blobs = Vec::with_capacity(self.per_core.len());

        for (core, args) in self.per_core {
            validate_section_layout(&layout.writer, &args.writer, "writer")?;
            validate_section_layout(&layout.reader, &args.reader, "reader")?;
            validate_section_layout(&layout.compute, &args.compute, "compute")?;

            let writer = lower_values(&args.writer);
            let reader = lower_values(&args.reader);
            let compute = lower_values(&args.compute);
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
            writer_patches: writer_patches.into(),
            reader_patches: reader_patches.into(),
            compute_patches: compute_patches.into(),
            blobs,
        };

        Ok((runtime_args, writer_args, reader_args, compute_args))
    }
}

impl RuntimeArgs {
    fn empty() -> Self {
        Self {
            cores: Arc::from([]),
            writer_patches: Arc::from([]),
            reader_patches: Arc::from([]),
            compute_patches: Arc::from([]),
            blobs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeArgPatchGroup {
    index: usize,
    offset: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PerCoreRuntimeArgs {
    writer: Vec<Option<u32>>,
    reader: Vec<Option<u32>>,
    compute: Vec<Option<u32>>,
}

fn lower_values(args: &[Option<u32>]) -> Vec<u32> {
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            Some(value) => values.push(*value),
            None => values.push(0),
        }
    }
    values
}

fn section_patches(args: &[Option<u32>], base_offset: usize) -> Vec<RuntimeArgPatchGroup> {
    args.iter()
        .enumerate()
        .filter_map(|(index, arg)| {
            arg.is_none().then_some(RuntimeArgPatchGroup {
                index,
                offset: base_offset + index * size_of::<u32>(),
            })
        })
        .collect()
}

fn validate_section_layout(
    expected: &[Option<u32>],
    actual: &[Option<u32>],
    section: &str,
) -> io::Result<()> {
    let matches = expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| expected.is_none() == actual.is_none());
    if matches {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{section} runtime arg layout differs across cores"
        )))
    }
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
