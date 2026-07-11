import os

replacements = {
    '->': '->',
    '=': '=',
    '[OK]': '[OK]',
    '[FAIL]': '[FAIL]',
    '[WARN]': '[WARN]',
    '-': '-',
    '"': '"',
    '"': '"',
    ''': "'"
}

for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.py') or f.endswith('.ps1'):
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                modified = False
                for k, v in replacements.items():
                    if k in content:
                        content = content.replace(k, v)
                        modified = True
                
                new_content = []
                for char in content:
                    if ord(char) > 127:
                        new_content.append('?')
                        modified = True
                    else:
                        new_content.append(char)
                
                if modified:
                    new_content_str = ''.join(new_content)
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content_str)
                    print(f'Fixed encoding in {filepath}')
            except Exception as e:
                pass
