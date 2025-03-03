const fs = require('fs');

try {
  const data = fs.readFileSync('./package.json', { encoding: 'utf-8' });
  const jsonData = JSON.parse(data);
  fs.writeFileSync('./version.json', JSON.stringify({ version: jsonData.version }), {
    encoding: 'utf-8',
  });
  console.log('\x1b[32m%s\x1b[0m', '版本号写入成功');
} catch (error) {
  console.error(`无法读取 package.json:`, error);
}
