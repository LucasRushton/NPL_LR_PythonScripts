import pexpect
from pexpect import popen_spawn
import pathlib
import re

magv_exe = r"C:\\Users\\m73377\\OneDrive - Microchip Technology Inc\\Documents\\Code\\MAG-V\\Console\\bin\\Debug\\net6.0\\magv.exe"
magv_exe = pathlib.PureWindowsPath(magv_exe).as_posix()

t = pexpect.popen_spawn.PopenSpawn(magv_exe)
while 1:
    t.expect("magv: ")
    t.sendline("vcsel read temp")
    print(re.sub(pattern="\\\\[a-z]|'|,|[a-z]", repl='', string=str(t.before)))

