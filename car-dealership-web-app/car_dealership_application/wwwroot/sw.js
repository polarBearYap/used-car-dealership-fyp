// Design pattern derived from https://web.dev/offline-cookbook/

self.addEventListener('install', function (event) {
    event.waitUntil(
        caches.open('car-dealership-static-v1').then(function (cache) {
            // Static files for the web app
            return cache.addAll([
                "/favicon.ico",
                "/css/default_style.min.css",
                "/css/site.css",
                "/font/font-awesome/css/font-awesome.min.css",
                "/font/font-awesome/fonts/fontawesome-webfont.eot",
                "/font/font-awesome/fonts/fontawesome-webfont.svg",
                "/font/font-awesome/fonts/fontawesome-webfont.ttf",
                "/font/font-awesome/fonts/fontawesome-webfont.woff",
                "/font/font-awesome/fonts/fontawesome-webfont.woff2",
                "/font/font-awesome/fonts/FontAwesome.otf",
                "/lib/chart-js/chart.esm.min.js",
                "/lib/chart-js/chart.min.js",
                "/lib/chart-js/helpers.esm.min.js",
                "/lib/jquery/jquery.min.js",
                "/lib/jquery/jquery.min.map",
                "/lib/jquery-validation/additional-methods.min.js",
                "/lib/jquery-validation/jquery-validation-sri.json",
                "/lib/jquery-validation/jquery.validate.min.js",
                "/lib/jquery-validation-unobtrusive/jquery.validate.unobtrusive.min.js",
                "/lib/twitter-bootstrap/css/bootstrap.min.css",
                "/lib/twitter-bootstrap/js/bootstrap.bundle.min.js",
                "/logo/logo_primary.png",
                "/js/app.bundle.js"
            ]);
        }),
    );
});

self.addEventListener('fetch', function (event) {
    // Parse the URL:
    const requestURL = new URL(event.request.url);
    const requestMethod = event.request.method;
    const requestPath = requestURL.pathname;

    if (requestMethod === 'GET') {
        // Routing for in-house Car Dealership Flask API
        if (requestURL.hostname === 'localhost' || requestURL.hostname === 'car-dealership-api.com') {
            console.log(/^api\/v\d\/car_[^\s]+\/global_feature_importance$/.test(requestPath));
            console.log(requestPath);
            // Handle /api/v1/car_price/global_feature_importance | /api/v1/car_demand/global_feature_importance and so on...
            if (/^\/api\/v\d\/car_[^\s]+\/global_feature_importance$/.test(requestPath)) {
                // Fetch resources from cache, if not found then fetch from server
                event.respondWith(
                    caches.open(`car-dealership-${requestPath.toLowerCase().replaceAll('/', '-')}`).then(function (cache) {
                        return cache.match(event.request).then(function (response) {
                            var fetchPromise = fetch(event.request).then(function (networkResponse) {
                                cache.put(event.request, networkResponse.clone());
                                return networkResponse;
                            });
                            return response || fetchPromise;
                        });
                    }),
                );
            }
        }

        // Routing for static files
        if (requestURL.origin === location.origin) {
            // Handle /css/... | /font/... | /js/... | /lib/... | /logo/... | /favicon.ico
            if (/^(?:~*)\/(?:css|font|js|lib|logo)\/|^\/favicon.ico/.test(requestPath)) {
                // Fetch resources from cache, if not found then fetch from server
                event.respondWith(
                    caches.match(event.request).then(function (response) {
                        return response || fetch(event.request);
                    }),
                );
            }
        }
    }
});

// Activate: Only use it for things you couldn't do while the old version was active.
self.addEventListener('activate', function (event) {
    event.waitUntil(
        // Perform clean up and migration
        caches.keys().then(function (cacheNames) {
            const STATIC_FILE_CACHE = 'car-dealership-static';
            const PRICE_API_CACHE = 'car-dealership-api-car_price-feature_importance';
            let cache_versions = {}
            cache_versions[STATIC_FILE_CACHE] = Array();
            cache_versions[PRICE_API_CACHE] = Array();

            // Find the latest version of static files cache
            for (let cacheName of cacheNames) {
                if (cacheName.startsWith(STATIC_FILE_CACHE)) {
                    cache_versions[STATIC_FILE_CACHE].push(cacheName.slice(-1,));
                }
                else if (cacheName.startsWith('car-dealership-api') && cacheName.endsWith('-car_price-feature_importance')) {
                    cache_versions[PRICE_API_CACHE].push(cacheName.slice(20)[0]);
                }
            }

            let latest_cache = [];

            if (cache_versions[STATIC_FILE_CACHE].length == 0) {
                latest_cache.push(`car-dealership-static-v${cache_versions[STATIC_FILE_CACHE].sort((a, b) => b - a)[0]}`);
            }
            if (cache_versions[PRICE_API_CACHE].length == 0) {
                latest_cache.push(`car-dealership-api-v${cache_versions[PRICE_API_CACHE].sort((a, b) => b - a)[0]}-car_price-feature_importance`);
            }

            // Remove the outdated caches
            return Promise.all(
                cacheNames
                    .filter(function (cacheName) {
                        !(latest_cache.includes(cacheName));
                    })
                    .map(function (cacheName) {
                        return caches.delete(cacheName);
                    }),
            );
        }),
    );

    // The rest of the URLs is fetched from remote servers
});