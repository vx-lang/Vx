use std::fs;
use vxc::gid::TypeId;
use vxc::metadata::VxMetadata;

#[test]
fn test_zero_copy_metadata_serialization() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_module.vxm");

    // 1. Generate a large synthetic dictionary of 10,000 TypeIds (320 KB)
    let mut original_dict = Vec::new();
    for i in 0..10000 {
        original_dict.push(TypeId {
            words: [i as u64, (i * 2) as u64, (i * 3) as u64, (i * 4) as u64],
        });
    }

    // 2. Save to file
    VxMetadata::save_to_file(&original_dict, &file_path).expect("Failed to save metadata");

    // 3. Ensure file exists and has the exact size
    // 8 bytes (len) + 10000 * 32 bytes (TypeId) = 320,008 bytes
    let metadata_len = fs::metadata(&file_path).unwrap().len();
    assert_eq!(metadata_len, 320_008);

    // 4. Load from file (Zero Copy)
    let buffer = fs::read(&file_path).expect("Failed to read metadata");
    let loaded_metadata = VxMetadata::load_from_buffer(&buffer);

    // 5. Verify integrity
    assert_eq!(loaded_metadata.type_dictionary.len(), original_dict.len());
    assert_eq!(loaded_metadata.type_dictionary, original_dict.as_slice());

    // Verify AST bytes are empty since we didn't add any
    assert_eq!(loaded_metadata.ast_data.len(), 0);

    // 6. Cleanup
    let _ = fs::remove_file(&file_path);
}
