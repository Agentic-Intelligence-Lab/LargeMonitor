"""Run DINO buffer monitoring on Split-CIFAR100 (online_lora split)."""

from lvmonitor.disjoint_dino import main

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("cifar100")
    main()
