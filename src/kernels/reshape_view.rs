use crate::dram::{tiled_allocation_shape, TILE_C, TILE_R};
use std::io;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct ReshapeSourceView {
    source_shape: Vec<usize>,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) tile_rows: u32,
    pub(crate) tiles_per_row: u32,
}

pub(crate) fn reshape_source_view(
    source_shape: &[usize],
    logical_shape: &[usize],
    name: &str,
) -> io::Result<ReshapeSourceView> {
    validate_same_volume(source_shape, logical_shape, name)?;
    let allocation_shape = tiled_allocation_shape(source_shape)?;
    let rank = allocation_shape.len();
    let (rows, cols) = matrix_dims(source_shape);
    Ok(ReshapeSourceView {
        source_shape: source_shape.to_vec(),
        rows: u32_value(rows, name)?,
        cols: u32_value(cols, name)?,
        tile_rows: u32_value(allocation_shape[rank - 2] / TILE_R, name)?,
        tiles_per_row: u32_value(allocation_shape[rank - 1] / TILE_C, name)?,
    })
}

pub(crate) fn optional_reshape_source_view(
    source_shape: Option<&[usize]>,
    logical_shape: &[usize],
    name: &str,
) -> io::Result<Option<ReshapeSourceView>> {
    source_shape
        .map(|source_shape| reshape_source_view(source_shape, logical_shape, name))
        .transpose()
}

impl ReshapeSourceView {
    pub(crate) fn source_shape(&self) -> &[usize] {
        &self.source_shape
    }
}

fn matrix_dims(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        rank => (shape[rank - 2], shape[rank - 1]),
    }
}

fn validate_same_volume(lhs: &[usize], rhs: &[usize], name: &str) -> io::Result<()> {
    let lhs_count = element_count(lhs)
        .ok_or_else(|| invalid_input(format!("{name} lhs element count overflow")))?;
    let rhs_count = element_count(rhs)
        .ok_or_else(|| invalid_input(format!("{name} rhs element count overflow")))?;
    if lhs_count != rhs_count {
        return Err(invalid_input(format!(
            "{name} requires same element count, got {lhs:?} and {rhs:?}"
        )));
    }
    Ok(())
}

fn element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn u32_value(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} value {value} does not fit in u32")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
