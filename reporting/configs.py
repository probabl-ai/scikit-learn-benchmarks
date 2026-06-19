
HARDWARE_NAMES = {
    "0611c8": "small-laptop"
}


def read_env(kind: str, hash: str):
    assert kind in ['software', 'hardware']
    # TODO

def summarize_software_env(env: dict, implementation: dict):
    # return a small dict, ready for use in templating
    # with relevant information in the env for the given implementation:
    # include:
    # - env name
    # - implementation["library"] version, and versions of important dependencies
    # - versions of array library
    # - if relevant: summary of BLAS/threading infos
    pass # TODO

def summarize_hardware_env(env: dict):
    # same for hardware, but independant of implem
    pass

