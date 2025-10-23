from pathlib import Path

# Read did_analysis.py
did_file = Path('src/03_analysis/did_analysis.py')
content = did_file.read_text()

# Find the section that creates event study variables
# We need to ensure event_time is created properly

# Add event_time creation if missing
if 'event_time' not in content or "event_" not in content:
    # Find where we load the data
    import_section = content.find('panel = pd.read_csv')
    if import_section != -1:
        # Find the end of data loading section
        next_section = content.find('\n\n', import_section)
        
        # Insert event_time creation code
        event_code = '''
    # Create event time variable
    panel['event_time'] = panel['year'] - panel['digital_year']
    panel['event_time'] = panel['event_time'].fillna(-999)  # For never-treated
    
    # Create event study dummies
    for k in range(-5, 6):  # -5 to +5
        if k < 0:
            panel[f'event_{k}'] = (panel['event_time'] == k).astype(int)
        elif k > 0:
            panel[f'event_{k}'] = (panel['event_time'] == k).astype(int)
    
    # Binned endpoints
    panel['event_pre'] = (panel['event_time'] < -5).astype(int)
    panel['event_post'] = (panel['event_time'] > 5).astype(int)
'''
        
        content = content[:next_section] + event_code + content[next_section:]

# Write back
did_file.write_text(content)
print("✓ Fixed did_analysis.py")
