import re

with open("src/codegen/generator.rs", "r") as f:
    content = f.read()

# Replace LowerToMelior::lower(s, self, block) with LowerToMelior::lower(s, self, region, block)
content = re.sub(r"LowerToMelior::lower\((.*?),\s*self,\s*block\)", r"LowerToMelior::lower(\1, self, region, block)", content)

with open("src/codegen/generator.rs", "w") as f:
    f.write(content)
