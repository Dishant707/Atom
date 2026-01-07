import sys
import urllib.request
import gzip
import shutil
import os

def download_file(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    gz_filename = f"{pdb_id}.cif.gz"
    cif_filename = f"{pdb_id}.cif"
    
    print(f"Python: Downloading {url}...")
    try:
        # User agent is sometimes required
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req) as response, open(gz_filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        print(f"Python: Downloaded {gz_filename}. Decompressing...")
        
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(cif_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        print(f"Python: Successfully created {cif_filename}")
        return 0
    except Exception as e:
        print(f"Python: Error - {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_helper.py <PDB_ID>")
        sys.exit(1)
    
    pdb_id = sys.argv[1]
    sys.exit(download_file(pdb_id))
