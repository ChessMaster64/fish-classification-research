from ptflops import get_model_complexity_info


def print_flops(model):

    macs, params = get_model_complexity_info(
        model,
        (3, 224, 224),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )

    print()

    print("=" * 40)
    print("Model Complexity")
    print("=" * 40)

    print(f"MACs       : {macs}")
    print(f"Parameters : {params}")

    print("=" * 40)

    print()