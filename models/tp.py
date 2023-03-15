import os

path = os.path.join(os.getcwd(), "ppo_shortened_30")
files = os.listdir(path)

for f in files:
    l = f.split('_')
    os.rename(os.path.join(path,f), os.path.join(path,l[-1]))

