import argparse
import json
import bz2


def printAllele_annotations(primary_refsnp):
    '''
    rs clinical significance
    '''
    for annot in primary_refsnp['allele_annotations']:
        for clininfo in annot['clinical']:
            print(",".join(clininfo['clinical_significances']))


def printPlacements(info):
    '''
    rs genomic positions
    '''

    for alleleinfo in info:
        # has top level placement (ptlp) and assembly info
        placement_annot = alleleinfo['placement_annot']
        if alleleinfo['is_ptlp'] and \
                len(placement_annot['seq_id_traits_by_assembly']) > 0:
            assembly_name = placement_annot[
                'seq_id_traits_by_assembly'][0]['assembly_name']

            for a in alleleinfo['alleles']:
                spdi = a['allele']['spdi']
                if spdi['inserted_sequence'] != spdi['deleted_sequence']:
                    (ref, alt, pos, seq_id) = (spdi['deleted_sequence'],
                                               spdi['inserted_sequence'],
                                               spdi['position'],
                                               spdi['seq_id'])
                    break
            print("\t".join([assembly_name, seq_id, str(pos), ref, alt]))


parser = argparse.ArgumentParser(description='Example of parsing '
                                             'JSON RefSNP Data')
parser.add_argument('-i', dest='input_fn', required=True,
                    help='The name of the input file to parse')

args = parser.parse_args()


cnt = 0
with bz2.BZ2File(args.input_fn, 'rb') as f_in:
    for line in f_in:
        rs_obj = json.loads(line.decode('utf-8'))
        print(rs_obj['refsnp_id'] + "\t")  # rs ID

        if 'primary_snapshot_data' in rs_obj:
            primary_snapshot_data = rs_obj['primary_snapshot_data']
            printPlacements(primary_snapshot_data['placements_with_allele'])
            printAllele_annotations(primary_snapshot_data)
            print("\n")

        cnt += 1
        if (cnt > 1000):
            break