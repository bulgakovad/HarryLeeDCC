from harry_lee_functions import compare_strfun


def main():
    """
    Prompt the user and call compare_strfun for one or many Q² bins.
    """
    mode = input("Choose option: 'compare_strfun' ").strip().lower()
    if mode != "compare_strfun":
        print("Only 'compare_strfun' is supported.")
        return
 
    try:
        q2_vals = [
            float(token) for token in
            input("Enter Q² values (GeV²) separated by commas: ").split(",")
            if token.strip()
        ]
        beam_energy = float(input("Enter beam energy E (GeV): "))
    except ValueError:
        print("Numbers only, please.")
        return

    for q2 in q2_vals:
        print(f"\n→ Processing Q² = {q2} GeV² …")
        try:
            compare_strfun(q2, beam_energy,
                           interp_file="input_data/wempx.dat",
                           onepi_file="input_data/wemp-pi.dat",
                           num_points=200)
        except Exception as err:
            print(f"  Error: {err}")


if __name__ == "__main__":
    main()
