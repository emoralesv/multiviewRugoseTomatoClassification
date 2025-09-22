import experimenterML.Experiment as exp
import experimenterML.dashboard as dash
import sys

def main():
    experiment = exp.Experimenter(r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\experiment.yaml")
    experiment.run()
    
    

if __name__ == "__main__":
    main()