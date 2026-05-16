use crate::gid::{deserialize_metadata_symbols, serialize_metadata_symbols, TypeId};
use std::fs;
use std::io;
use std::path::Path;

/// Represents the serialized metadata dictionary for a specific module/crate.
pub struct VxMetadata<'a> {
    /// The zero-copy slice of 256-bit TypeIds directly mapped from the file buffer
    pub type_dictionary: &'a [TypeId],
    /// The remaining variable-length AST data (signatures, etc)
    pub ast_data: &'a [u8],
}

impl<'a> VxMetadata<'a> {
    /// Save the fully deduplicated Type Dictionary to disk.
    pub fn save_to_file(type_dictionary: &[TypeId], path: &Path) -> io::Result<()> {
        let mut buffer = Vec::new();
        // Zero-copy serialization via bytemuck
        serialize_metadata_symbols(type_dictionary, &mut buffer);

        // For now, AST bytes are empty since we're just saving the dictionary
        let ast_bytes: Vec<u8> = Vec::new();
        buffer.extend_from_slice(&ast_bytes);

        fs::write(path, &buffer)?;
        Ok(())
    }

    /// Load the Type Dictionary directly from disk using a Zero-Copy memory cast.
    /// This requires the caller to own the `[u8]` backing buffer, returning
    /// a struct containing slice references mapped over that buffer.
    pub fn load_from_buffer(buffer: &'a [u8]) -> Self {
        let (type_dictionary, ast_data) = deserialize_metadata_symbols(buffer);
        Self {
            type_dictionary,
            ast_data,
        }
    }
}
