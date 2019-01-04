import os
import sys

# shameless hack, but it works conveniently
os.system("pip install -r requirements.txt")

os.chdir("apex")

os.system("python " + " ".join(sys.argv) )

