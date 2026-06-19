import os

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content.replace('CPU_AVX512', 'CpuAvx512').replace('CPU_Neon', 'CpuNeon')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def main():
    for root, _, files in os.walk('/Users/adityak/go/Vx/src'):
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                replace_in_file(filepath)

if __name__ == "__main__":
    main()
