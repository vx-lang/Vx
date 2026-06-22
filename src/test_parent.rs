use melior::{Context, ir::{Block, Region}};
fn main() {
    let context = Context::new();
    let region = Region::new();
    let block = region.append_block(Block::new(&[]));
    let parent = block.parent();
    println!("Has parent: {}", parent.is_some());
}
