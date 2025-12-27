import argparse

def main():
    parser = argparse.ArgumentParser(prog='oracle', description='Protein solubility predictor')
    subparsers = parser.add_subparsers(dest='command')
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('sequence')
    predict_parser.add_argument('--visualize', '-v', action='store_true')
    predict_parser.add_argument('--minimal', '-m', action='store_true')
    
    subparsers.add_parser('info')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        from oracle import Oracle
        predictor = Oracle()
        result = predictor.predict(args.sequence)
        print(result.summary())
        if args.visualize:
            result.visualize('minimal' if args.minimal else 'full')
    elif args.command == 'info':
        from oracle import Oracle
        predictor = Oracle(verbose=False)
        info = predictor.model_info
        print(f"\nORACLE - Protein Solubility Predictor")
        print(f"Training Samples: {info.get('training_samples', 'N/A'):,}")
        print(f"Test Accuracy: {info.get('test_accuracy', 0):.1%}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
