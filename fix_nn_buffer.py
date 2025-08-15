#!/usr/bin/env python3
"""
Fix nn.Buffer -> register_buffer in HRM code
"""

import os

def fix_file(filepath):
    """Fix nn.Buffer usage in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if file needs fixing
    if 'nn.Buffer' not in content:
        return False
    
    # Replace nn.Buffer with proper register_buffer calls
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'nn.Buffer(' in line:
            # Extract variable name
            if '=' in line:
                var_part = line.split('=')[0].strip()
                var_name = var_part.split('.')[-1].strip()
                
                # Replace nn.Buffer with register_buffer
                new_line = line.replace('nn.Buffer(', f'self.register_buffer(\'{var_name}\', ')
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"✓ Fixed {filepath}")
    return True

# Fix the files
files_to_fix = [
    '/home/dp/ai-workspace/HRM/models/sparse_embedding.py',
    '/home/dp/ai-workspace/HRM/models/hrm/hrm_act_v1.py'
]

for file in files_to_fix:
    if os.path.exists(file):
        fix_file(file)

print("\n✅ All files fixed!")