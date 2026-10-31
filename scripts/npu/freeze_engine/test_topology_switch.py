"""Refuse changed ownership and process identity before switching topology."""
import copy
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import topology_switch as m

class IdentityTest(unittest.TestCase):
    def test_identity_changes_refused(self):
        c={'name':'saea-real500-tp4-i0','profile_id':'qwen3-omni-real500-2xtp4-s96-b2048-v1'}
        good={'Name':'/'+c['name'],'Id':'original','Image':m.common.IMAGE,'Config':{'Labels':{m.common.LABEL:c['profile_id']}},'State':{'Running':True},'RestartCount':0,'HostConfig':{'RestartPolicy':{'Name':'no'}}}
        m.validate(good,c,'original')
        for key,value in [('Id','replacement'),('RestartCount',1),('Image','foreign'),('Config',{'Labels':{}}),('State',{'Running':False})]:
            changed=copy.deepcopy(good);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate(changed,c,'original')

class RecoveryTest(unittest.TestCase):
    def test_only_exact_stopped_owned_archive_is_accepted(self):
        c={'name':'old','profile_id':'owned'}
        good={'Name':'/archive','Id':'original','Image':m.common.IMAGE,
              'Config':{'Labels':{m.common.LABEL:'owned'}},'State':{'Running':False},'RestartCount':0}
        m.validate_stopped_archive(good,c,'original','archive')
        for key,value in [('Id','new'),('Name','/wrong'),('Image','foreign'),
                          ('Config',{'Labels':{}}),('State',{'Running':True}),('RestartCount',1)]:
            changed=copy.deepcopy(good);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.validate_stopped_archive(changed,c,'original','archive')

class ReleaseTest(unittest.TestCase):
    def test_delayed_memory_release_waits_without_recreation(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m.common,'call',return_value='smi'), patch.object(m,'commands',side_effect=[{'ready':False,'memory_release_only':True},{'ready':True}]) as inspect, patch.object(m.time,'sleep') as sleep:
            m.wait_card_release('profile',Path(root))
            self.assertEqual(inspect.call_count,2);sleep.assert_called_once_with(10)
            self.assertEqual(len(list(Path(root).glob('cards-release-*'))),2)
    def test_process_or_health_refuses_immediately(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m.common,'call',return_value='smi'), patch.object(m,'commands',return_value={'ready':False,'memory_release_only':False}), patch.object(m.time,'sleep') as sleep:
            with self.assertRaises(ValueError):m.wait_card_release('profile',Path(root))
            sleep.assert_not_called()

if __name__=='__main__':unittest.main()
