from pathlib import Path
import re



TITLE_PTN = re.compile(r'^(?P<level>#+)\s*(?P<numbering>\d+(\.\d+)*\.)?\s+(?P<title>.*)$')

def number_headings(markdown_text):
    """
    Adds numbering to markdown headings. If headings are already numbered, it updates the numbering.
    
    Args:
    - markdown_text (str): The markdown text with headings

    Returns:
    - str: The markdown text with updated numbered headings

    Example:       
        Input:
            # 标题1
            ## 标题2
            ### 标题3
            #### 标题4
            ##### 标题5
            ### 标题6
            ## 标题7
            ### 标题8
            ### 标题9
        Output:
            # 标题1
            ## 1. 标题2
            ### 1.1. 标题3
            #### 1.1.1. 标题4
            ##### 1.1.1.1. 标题5
            ### 1.2. 标题6
            ## 2. 标题7
            ### 2.1. 标题8
            ### 2.2. 标题9
    """
    lines = markdown_text.split('\n')
    heading_count = [0] * 10  # Supports heading levels 1 to 9

    for i in range(len(lines)):
        line = lines[i]
        m = TITLE_PTN.match(line)

        if m is None:
            continue
        
        grp_dict = m.groupdict()
        level = len(grp_dict['level']) - 1
        

        title = grp_dict['title']

        # Count the number of '#' to determine the heading level
        heading_count[level] += 1
        # Reset all lower heading counts to 0
        for j in range(level + 1, 10):
            heading_count[j] = 0
        # Construct the numbering string
        numbering = '.'.join(str(heading_count[j]) for j in range(1, level + 1) if heading_count[j] > 0)

        # Add numbering to the heading, removing existing numbering if present
        lines[i] = '#' * (level+1) + ' ' + numbering + '. ' + title

    return '\n'.join(lines)

if __name__ == '__main__':
    
    # s1 = '##### 1.1.1.1. 标题5'
    # s2 = '# Report for AAI Project: Train classifier for modified MNIST with unstable (spurious) features'
    # print(TITLE_PTN.match(s1).groupdict())
    # print(TITLE_PTN.match(s2).groupdict())

    md_path = '../docs/project_report.md'

    content = Path(md_path).read_text(encoding='utf-8')
    numbered = number_headings(content)
    print(numbered)
