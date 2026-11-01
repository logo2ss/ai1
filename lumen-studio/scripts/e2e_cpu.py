"""Offline browser smoke test. Requires running engine with token t on port 8820.
Run with the project's venv after installing the optional playwright dev tool.
CHROMIUM may select a local Chromium executable (no browser download at runtime).
"""
import json
import math
import os
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

BASE='http://127.0.0.1:8820'
OUT=Path('validation'); OUT.mkdir(exist_ok=True)

def small(name):
    path=Path('/tmp/lumen-e2e')/Path(name).name; path.parent.mkdir(exist_ok=True)
    im=Image.open(Path('testdata')/name).convert('RGB');im.thumbnail((512,512));im.save(path)
    return str(path)

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM','/usr/bin/chromium'),headless=True,args=['--disable-gpu','--disable-software-rasterizer','--no-sandbox'])
    page=browser.new_page(viewport={'width':1550,'height':1050})
    errors=[]; external=[]; completed=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    def route(r):
        if r.request.url.startswith((BASE+'/', 'blob:', 'data:')): r.continue_()
        else: external.append(r.request.url);r.abort()
    page.route('**/*',route)
    def run_button(name,endpoint):
        button=page.get_by_role('button',name=name,exact=True);button.wait_for(timeout=120000)
        page.wait_for_function('(text) => [...document.querySelectorAll("button")].some(b => b.textContent.trim()===text && !b.disabled)',arg=name,timeout=120000)
        with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith(endpoint),timeout=120000) as response: button.click()
        r=response.value;assert r.ok,r.text();job=r.json()['job_id']
        for _ in range(900):
            j=page.request.get(f'{BASE}/api/jobs/{job}?token=t').json()
            if j['state'] in ('done','error'):break
            page.wait_for_timeout(200)
        assert j['state']=='done',j
        print('PASS', name, endpoint, flush=True)
        return j['result']
    def frame(testid):
        page.get_by_test_id(testid).wait_for(timeout=120000)
        page.wait_for_function('(testid) => { const e=document.querySelector(`[data-testid="${testid}"] [data-testid="cpu-frame"]`);return e && e.naturalWidth>0 && getComputedStyle(e).visibility!=="hidden"; }',arg=testid,timeout=120000)
        assert page.get_by_test_id(testid).get_attribute('data-renderer')=='cpu'
    page.goto(BASE+'/?token=t')
    assert page.evaluate("document.createElement('canvas').getContext('webgl') === null")
    page.get_by_test_id('nav-face').click()
    page.locator('input[type=file]').nth(0).set_input_files(small('faces/biden.jpg'))
    page.locator('input[type=file]').nth(1).set_input_files(small('faces/obama.jpg'))
    result=run_button('Replace face','/api/face/swap');completed.append('Natural face swap')
    assert page.request.get(f"{BASE}/api/images/{result['image_id']}/export?format=png&token=t").ok
    page.screenshot(path=str(OUT/'face-no-webgl.png'))
    page.get_by_test_id('nav-region').click()
    page.locator('input[type=file]').nth(0).set_input_files(small('scenes/fruits.jpg'))
    page.locator('input[type=file]').nth(1).set_input_files(small('scenes/home.jpg'))
    for label in ('IMAGE A','IMAGE B'):
        canvas=page.get_by_test_id('canvas-'+label);canvas.wait_for();box=canvas.bounding_box()
        for i in range(12):
            a=2*math.pi*i/12;page.mouse.click(box['x']+box['width']*(0.5+0.18*math.cos(a)),box['y']+box['height']*(0.5+0.18*math.sin(a)))
    result=run_button('Swap','/api/region/swap');assert set(result)=={'a','b'};completed.append('Region swap both ways from drawn selections')
    page.get_by_test_id('result-a').wait_for(timeout=120000);page.screenshot(path=str(OUT/'region-no-webgl.png'))
    page.get_by_test_id('nav-recon').click()
    assert page.get_by_test_id('recon-tool').get_by_role('button').all_text_contents()==['3D scene','Body 3D','Body proportions']
    assert page.get_by_role('button',name='Person 3D',exact=True).count()==0
    page.locator('input[type=file]').first.set_input_files(small('scenes/home.jpg'))
    result=run_button('Reconstruct 3D scene','/api/recon');frame('viewer');completed.append('Scene CPU reconstruction/render')
    viewer=page.get_by_test_id('viewer');box=viewer.bounding_box();page.mouse.move(box['x']+box['width']*.9,box['y']+box['height']*.5);page.mouse.down();page.mouse.move(box['x']+box['width']*.1,box['y']+box['height']*.5,steps=20);page.mouse.up();page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT/'scene-no-webgl.png'))
    page.locator('input[type=file]').first.set_input_files(small('faces/obama.jpg'))
    page.get_by_role('button',name='Body 3D',exact=True).click();run_button('Build 3D body model','/api/body3d');frame('body-viewer');completed.append('Body 3D CPU build/render')
    page.get_by_role('button',name='Body proportions',exact=True).click()
    width=page.get_by_role('slider',name='Overall width',exact=True);width.wait_for(timeout=120000)
    width.focus();page.keyboard.press('ArrowRight');page.keyboard.press('ArrowRight')
    run_button('Apply & naturalise','/api/body/apply')
    assert page.locator('body').inner_text().find('Engine unavailable')<0
    page.screenshot(path=str(OUT/'body-proportions-no-webgl.png'));completed.append('Body proportions detection, preview and apply')
    page.get_by_test_id('nav-models').click();page.get_by_test_id('nav-settings').click()
    assert not errors,errors
    assert not external,external
    (OUT/'e2e-cpu.json').write_text(json.dumps({'passed':completed,'page_errors':errors,'external_requests':external,'webgl':False},indent=2))
    print(json.dumps(completed));browser.close()
