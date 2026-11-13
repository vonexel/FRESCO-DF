import numpy as np
import pandas as pd
import pytest
from fresco.synthetic import make_cache
from fresco.training import Coordinates, train_mask, build_model
from fresco.run import load_cache
from fresco.utils import atomic_json,digest_file


def test_target_changes_cannot_change_fitted_coordinates(tmp_path):
    cache=make_cache(tmp_path/"cache")
    f,c,m=load_cache(cache)
    a=Coordinates().fit(f,c,m,8,True)
    changed=np.array(f)
    changed[m.dataset=="celebdfpp"]+=1000
    altered=np.array(c)
    altered[m.dataset=="celebdfpp"]-=1000
    b=Coordinates().fit(changed,altered,m,8,True)
    np.testing.assert_array_equal(a.components,b.components)
    np.testing.assert_array_equal(a.cmean,b.cmean)
    assert set(m.loc[train_mask(m),"dataset"]) == {"ffpp"}


def test_target_training_and_duplicate_frame_cache_rejected(tmp_path):
    cache=make_cache(tmp_path/"cache")
    m=pd.read_csv(cache/"metadata.csv")
    m.loc[m.dataset=="celebdfpp","split"]="train"
    m.to_csv(cache/"metadata.csv",index=False)
    import json
    manifest=json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
    manifest["metadata_sha256"]=digest_file(cache/"metadata.csv")
    atomic_json(cache/"manifest.json",manifest)
    with pytest.raises(ValueError,match="TEST ONLY"):
        load_cache(cache)


def test_full_feature_reference_retains_native_coordinates(tmp_path):
    cache=make_cache(tmp_path/"cache")
    f,c,m=load_cache(cache)
    transform=Coordinates().fit(f,c,m,f.shape[1],False)
    np.testing.assert_array_equal(transform.components,np.eye(f.shape[1]))
    x=transform.transform(f,c)
    assert x.shape==f.shape
    np.testing.assert_allclose(x[train_mask(m)].std(0),1,atol=1e-5)


def test_prespecified_primary_budget_can_be_achieved():
    _,kan=build_model('kan_spline',20,'params',72)
    _,gelu=build_model('mlp_gelu',20,'params',72)
    assert kan['feasible'] and gelu['feasible']
    assert abs(kan['params']/gelu['params']-1) <= .05


def test_manifest_rejects_finite_corrupted_feature_content(tmp_path):
    cache=make_cache(tmp_path/"cache")
    atomic_json(cache/"manifest.json",{"features_sha256":digest_file(cache/"features.npy")})
    data=np.load(cache/"features.npy",mmap_mode="r+")
    data[0,0]+=.01
    data.flush()
    del data
    with pytest.raises(ValueError,match="Corrupted features cache"):
        load_cache(cache)

