import sys
import pprint
from funlib.geometry import Coordinate
from volara.blockwise import ExtractFrags
from volara.blockwise import AffAgglom
from volara.blockwise import GlobalMWS
from volara.blockwise import LUT
from volara.datasets import Affs, Labels
from volara.dbs import SQLite, PostgreSQL


def run_blockwise_mws(
        in_f,
        in_ds,
        out_f,
        out_frags,
        out_seg
):

    # Configure your db
    db = SQLite(
        path=f"/tmp/db.sqlite",
        edge_attrs={
            "zyx_aff": "float",
        },
    )

    # Configure your arrays
    affinities = Affs(
        store=f"{in_f}/{in_ds}",
        neighborhood=[
            Coordinate(1, 0, 0), 
            Coordinate(0, 1, 0), 
            Coordinate(0, 0, 1), 
            Coordinate(2, 0, 0),
            Coordinate(0, 5, 0), 
            Coordinate(0, 0, 5), 
        ],
    )
    fragments = Labels(store=f"{out_f}/{out_frags}")

    ## Extract Fragments
    extract_frags = ExtractFrags(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        block_size=(20, 100, 100),
        context=(2, 2, 2),
        num_workers=10,
        bias=[-0.5, -0.5, -0.5, -0.8, -0.8, -0.8],
        # sigma=[1, 2, 2],
        # noise_eps=0.01,
    )

    extract_frags.drop() # to restart + remove logs cahce
    extract_frags.run_blockwise(multiprocessing=True)

    # Affinity Agglomeration across blocks
    aff_agglom = AffAgglom(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        block_size=(20, 100, 100),
        context=(2, 2, 2),
        num_workers=20,
        scores={"zyx_aff": affinities.neighborhood},
    )
    aff_agglom.run_blockwise(multiprocessing=True)

    # Global MWS
    global_mws = GlobalMWS(
        db=db,
        frags_data=fragments,
        lut=f"{out_f}/lut",
        bias={"zyx_aff": -0.5},
    )
    global_mws.run_blockwise(multiprocessing=False)

    segments = Labels(store=f"{out_f}/{out_seg}")

    # Write LUT
    lut = LUT(
        frags_data=fragments,
        seg_data=segments,
        lut=f"{out_f}/lut",
        block_size=(20, 100, 100),
        num_workers=20
    )
    lut.run_blockwise(multiprocessing=True)


if __name__ == "__main__":
    in_f = sys.argv[1]
    in_ds = sys.argv[2]
    out_ds = sys.argv[3]

    out_f = in_f
    out_frags_ds = f'{out_ds}/frags'
    out_seg_ds = f'{out_ds}/seg'

    run_blockwise_mws(in_f,in_ds,out_f,out_frags_ds,out_seg_ds)