"""Opt-in fresh target-protection pilot; does not change the historical preset."""
import train_fresh_pilot as fresh
fresh.PROTECTED = True
fresh.OUT = fresh.ROOT / 'outputs/protected_pilot'
if __name__ == '__main__':
    fresh.main()