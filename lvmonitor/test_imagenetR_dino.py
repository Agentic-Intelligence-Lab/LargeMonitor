"""Run DINO buffer monitoring on Split-ImageNet-R (online_lora split)."""

from lvmonitor.disjoint_dino import main

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(["imagenetR", "--model-tag", "vits16"])
    main()
