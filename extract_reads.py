

'''
#20: 99% correct base call
#30: 99.9% correct base call
#40: 99.99% correct base call            
#source: https://www.illumina.com/documents/products/technotes/technote_Q-Scores.pdf
'''
import argparse
import gzip
import sys
import itertools

parser = argparse.ArgumentParser(description="extract reads from .fastq.gz")
parser.add_argument("-F", metavar="<input.fastq.gz>",help="input .fastq.gz file needs to be zipped")
parser.add_argument("-R", metavar="<read_number>", help="readnumber: how many reads to extract")
parser.add_argument("-O", metavar="<outputfile.txt>", help="write reads to output file")
parser.add_argument("-Q", metavar="<phred threshold>", help="default 30")
parser.add_argument("-L", metavar="<min read length>", help="default 140")
args = parser.parse_args()

if __name__ == "__main__":
    
    file_path = args.F if args.F is not None else "test.fastq.gz"
    rn = int(args.R) if args.R is not None else 100000
    outputfile_path = args.O if args.O is not None else "test_out.txt"
    phred_threshold = int(args.Q) if args.Q is not None else 30
    min_read_length = int(args.L) if args.L is not None else 140
    outputfile = open(outputfile_path, "w")
    
    count = 0    
    with gzip.open(file_path, 'rt') as f:                
        
        for l1, l2, l3, l4 in itertools.zip_longest(*[f]*4):                                    
            seq = l2
            seq_qc = [ord(char) - 33 for char in l4]            
            avg_quality = sum(seq_qc)/len(seq_qc)            
                                    
            if avg_quality > phred_threshold and len(seq) > min_read_length:
                count += 1
                outputfile.write(seq)                                

            if (count % rn == 0 and count > 0):                
                sys.exit(0)
                        
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
