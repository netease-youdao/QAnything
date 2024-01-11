const fs = require('fs');

try {
    const data = fs.readFileSync('./package.json', {encoding: 'utf-8'});
    const jsonData = JSON.parse(data);
    fs.writeFileSync('./version.txt', jsonData.version, {encoding: 'utf-8'})
} catch (error) {
    console.error(`无法读取 package.json:`, error);
}