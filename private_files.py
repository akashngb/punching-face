"""Owner-only permissions for the local secret stores: POSIX modes, or Windows ACLs."""
import os, subprocess
from pathlib import Path

def _account():
    # A SID is unambiguous where a bare user name can collide with a domain account.
    done=subprocess.run(['whoami','/user','/fo','csv','/nh'],capture_output=True,text=True)
    sid=done.stdout.strip().strip('"').split('","')[-1] if done.returncode==0 else ''
    return '*'+sid if sid.startswith('S-1-') else os.environ.get('USERNAME','')

def restrict(path):
    """Leave path reachable by this account alone. Raises OSError when that cannot be enforced."""
    path=Path(path)
    if os.name!='nt':path.chmod(0o700 if path.is_dir() else 0o600);return
    # chmod only toggles the read-only bit on Windows, so the mode is no guarantee at all.
    # Drop the inherited entries (SYSTEM, Administrators) and grant the owner full control.
    account=_account()
    if not account:raise OSError(f'Could not identify this account to restrict {path}.')
    grant=f'{account}:(OI)(CI)F' if path.is_dir() else f'{account}:F'
    done=subprocess.run(['icacls',str(path),'/inheritance:r','/grant:r',grant],capture_output=True,text=True)
    if done.returncode:raise OSError(f'Could not restrict {path}: '+(done.stderr.strip() or done.stdout.strip()))

def holders(path):
    """Who can reach path: a POSIX mode string, or the Windows identities named in its ACL."""
    path=Path(path)
    if os.name!='nt':return {oct(path.stat().st_mode&0o777)}
    out=subprocess.run(['icacls',str(path)],capture_output=True,text=True).stdout
    names=set()
    for line in out.splitlines():
        entry=line.replace(str(path),'',1).strip()
        if not entry or entry.startswith('Successfully'):break
        names.add(entry.rsplit(':',1)[0].strip().casefold())  # Windows account names are case-insensitive.
    return names

def owner_only():
    """The holders() value a correctly restricted path has on this platform."""
    if os.name!='nt':return {oct(0o600)}
    done=subprocess.run(['whoami'],capture_output=True,text=True)
    return {done.stdout.strip().casefold()}
