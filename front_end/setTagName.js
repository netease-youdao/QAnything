const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

const pkgPath = path.join(__dirname, './package.json');
const package = require(pkgPath);

const packInfo = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).replace(
  '\n',
  ''
);
const packInfoList = packInfo.split('/v');
const branchName = packInfoList[0];
const version = packInfoList[1];
setName();

function setName() {
  const blackName = ['test', 'master', 'release', 'stage', 'init'];
  if (blackName.indexOf(branchName) > -1) {
    return;
  }
  if (package.tagname !== branchName || package.version !== version) {
    let pkg = fs.readFileSync(pkgPath);
    pkg = JSON.parse(pkg);
    if (branchName && package.tagname !== branchName) {
      pkg.tagname = branchName;
    }
    if (version && package.version !== version) {
      pkg.version = version;
    }
    fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
  }
}
// "tagname":"ss",
