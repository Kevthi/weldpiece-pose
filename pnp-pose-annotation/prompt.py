import promptlib

prompter = promptlib.Files()

dir = prompter.dir()
file = prompter.file()

print(dir, '\n', file)
