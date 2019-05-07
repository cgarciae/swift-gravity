// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-gravity",
    products: [
        .executable(name: "Pure", targets: ["Pure"]),
        .executable(name: "Numpy", targets: ["Numpy"]),
        .executable(name: "TF", targets: ["TF"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/kylef/Commander.git", from: "0.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(name: "Pure", dependencies: ["Commander"], path: "Pure"),
        .target(name: "Numpy", dependencies: ["Commander"], path: "Numpy"),
        .target(name: "TF", dependencies: ["Commander"], path: "TF"),
    ]
)
