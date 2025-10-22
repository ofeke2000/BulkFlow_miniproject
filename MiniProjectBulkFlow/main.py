from src.data_loader import load_rockstar, load_cf4
from src.overdensity import find_origin
from src.masks import build_cf4_mask, build_uniform_mask
from src.bulkflow import compute_bulkflow
from src.experiment import run_experiments

def main():
    halos = load_rockstar()
    cf4 = load_cf4()

    origins = find_origin(halos)   # pick candidate origins δ5 ~ 0
    run_experiments(origins, halos, cf4)

if __name__ == "__main__":
    main()