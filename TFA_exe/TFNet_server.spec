# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
SETUP_DIR = '.\\'

a = Analysis(['TFNet_server.py',
                'complexModule.py',
                'complexLayers.py',
                'complexFunctions.py',
                'TFNet_model.py',
                'TFNet_util.py'],
             pathex=['.\\'],
             binaries=[],
             datas=[(SETUP_DIR+'model','model'),(SETUP_DIR+'config','config')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='TFNet_server',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='TFNet')
