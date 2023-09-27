def heading(text, width=80):
    print('\n' + '=' * width)
    print(text)
    print('=' * width)

def inline_heading(text, width=12):
    print('\n' + '#' * width + ' ' + text + ' ' + '#' * width)

def status_update(text):
    print('Status: ' + text, end='\r')
