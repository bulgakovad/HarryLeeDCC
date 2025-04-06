from functions import compute_cross_section, plot_cross_section_vs_W, generate_table, compare_strfun, compare_exp_model_pdf

def main():
    """
    Main routine: ask the user whether to calculate a single cross section ("calc"),
    plot cross sections ("plot"), generate a table ("table"), compare structure
    function data ("compare_strfun"), or compare experimental data with the PDF-based model ("compare_exp_model_pdf"),
    then perform the requested action.
    """
    mode = input("Enter 'calc'/'plot'/'table'/'compare_strfun'/'compare_exp_model_pdf': ").strip().lower()
    
    file_path = "input_data/wempx.dat"
    
    if mode == "calc":
        try:
            W_input = float(input("Enter W (GeV): "))
            Q2_input = float(input("Enter Q² (GeV²): "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        try:
            xsec = compute_cross_section(W_input, Q2_input, beam_energy, file_path=file_path, verbose=True)
            print("\n=== Results ===")
            print(f"Beam energy: {beam_energy:.5f} GeV")
            print(f"W: {W_input:.5f} GeV, Q²: {Q2_input:.5f} GeV²")
            print(f"Differential cross section dσ/dW/dQ²: {xsec:.5e} (10^(-30) cm²/GeV³)")
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "plot":
        try:
            Q2_input = float(input("Enter fixed Q² (GeV²) for plotting cross section vs W: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            plot_cross_section_vs_W(Q2_input, beam_energy, file_path=file_path, num_points=200)
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "table":
        try:
            fixed_Q2 = float(input("Enter fixed Q² (GeV²) for generating the table: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
            output_filename = input("Enter output filename (default: table.txt): ").strip()
            if output_filename == "":
                output_filename = "table.txt"
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            generate_table(file_path, fixed_Q2, beam_energy)
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "compare_strfun":
        try:
            fixed_Q2 = float(input("Enter fixed Q² (GeV²) for comparison: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            compare_strfun(fixed_Q2, beam_energy, interp_file=file_path, num_points=200)
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "compare_exp_model_pdf":
        try:
            Q2_input = float(input("Enter fixed Q² (GeV²) for PDF-based comparison: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            compare_exp_model_pdf(Q2_input, beam_energy, num_points=200)
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Invalid option. Please enter 'calc', 'plot', 'table', 'compare_strfun', or 'compare_exp_model_pdf'.")

if __name__ == "__main__":
    main()
