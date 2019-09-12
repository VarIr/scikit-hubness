# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
import os
import platform
import pytest
from skhubness.utils.io import create_tempfile_preferably_in_dir


@pytest.mark.parametrize('directory', [None, '/does/not/exist/kluawev'])
@pytest.mark.parametrize('persistent', [True, False])
def test_tempfile(directory, persistent):
    f = create_tempfile_preferably_in_dir(directory=directory, persistent=persistent)
    assert isinstance(f, str)
    if persistent and platform.system() != 'Windows':  # locked by running process on Windows
        os.remove(f)
