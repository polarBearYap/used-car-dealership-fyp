const path = require('path');
let webpack = require('webpack');

// Source: https://webpack.js.org/guides/typescript/
module.exports = {
    mode: 'development', // For development purpose only
    // Source: https://webpack.js.org/configuration/devtool/
    devtool: 'source-map', // For development purpose only
    // target: 'es5',
    entry: './Typescripts/main.ts',
    module: {
        rules: [
            {
                test: /\.ts$/,
                exclude: /node_modules/,
                loader: 'ts-loader'
            },
            // All output '.js' files will have any sourcemaps re-processed by 'source-map-loader'.
            {
                test: /\.js$/,
                loader: "source-map-loader"
            },
        ]
    },
    resolve: {
        modules: ['node_modules'],
        // Attempt to resolve these extensions in order.
        // If multiple files share the same name but have different extensions, webpack will resolve the one with the extension listed first in the array and skip the rest.
        extensions: [".webpack.js", ".web.js", ".ts", ".tsx", ".js"],
        fallback: {
            "timers": require.resolve("timers-browserify")
        }
    },
    output: {
        path: path.resolve(__dirname, 'wwwroot/js'),
        filename: 'app.bundle.js', 
        library: 'App',
        libraryTarget: 'var',
    },
    //optimization: {
    //    emitOnErrors: true
    //},
    stats: 'errors-warnings',
    plugins: [
        new webpack.ProvidePlugin({
            $: 'jquery',
            jQuery: 'jquery',
            jquery: 'jquery',
            bootstrap: 'bootstrap',
            'chart.js': 'chart.js'
        })
    ]
}