from functions import compute_cross_section, plot_cross_section_vs_W

def main():
    """
    Main routine: ask the user whether to calculate a single cross section or plot
    the cross section versus W, then perform the requested action.
    """
    mode = input("Enter 'calc' to calculate cross section or 'plot' to plot cross section vs W: ").strip().lower()
    
    if mode == "calc":
        try:
            W_input = float(input("Enter W (GeV): "))
            Q2_input = float(input("Enter Q² (GeV²): "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        try:
            xsec = compute_cross_section(W_input, Q2_input, beam_energy, file_path="input_data/wempx.dat", verbose=True)
            print("\n=== Results ===")
            print(f"Beam energy: {beam_energy:.5f} GeV")
            print(f"W: {W_input:.5f} GeV, Q²: {Q2_input:.5f} GeV²")
            print(f"Differential cross section dσ/dW/dQ²: {xsec:.5e} (10^(-30) cm²/GeV³)")
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "plot":
        try:
            Q2_input = float(input("Enter Q² (GeV²): "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            plot_cross_section_vs_W(Q2_input, beam_energy, file_path="input_data/wempx.dat")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Invalid option. Please enter 'calc' or 'plot'.")

if __name__ == "__main__":
    main()
