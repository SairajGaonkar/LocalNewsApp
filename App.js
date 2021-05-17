import { StatusBar } from "expo-status-bar";
import React, {useEffect} from "react";
import { StyleSheet, Text, View, Button } from "react-native";
//import Navbar from "./Components/Navbar.js";
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from './Components/HomeScreen.js';
import SecondScreen from './Components/SecondPage.js';
import LandingScreen from './Components/LandingScreen.js';
import SignInScreen from './Components/LoginScreen.js';
import SignUpScreen from './Components/SignUpScreen';
import ProfileScreen from './Components/ProfilePage';
import axios from 'axios';
import { Icon } from "react-native-elements";
import { set } from "react-native-reanimated";
import deviceStorage from "./Services/DeviceStorage.js";
//import { useEffect } from "react/cjs/react.production.min";
import AsyncStorage  from '@react-native-async-storage/async-storage';

let url='http://127.0.0.1:5000';
// let url='http://192.168.0.146:5000';
const RootStack = createStackNavigator();

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [loginError, setLoginError] = React.useState(false);
  const [regisError, setRegisError] = React.useState(false);
  
  useEffect(() =>{

    AsyncStorage.getItem('id_token', (err, result) => {
      if(result){
        setIsAuthenticated(true);
      }else{
        setIsAuthenticated(false);
      }
    });

  });
  const handleSignIn = (email, password) => {
    console.log('email:',email);
    console.log('password:',password);
    // TODO implement real sign in mechanism
    let payload = {
      email_id : email,
      pass : password
    };
    axios
        .post(url+"/login", payload, {
          headers: {
            "content-type": "application/json",
          },
        })
        .then((response) => {
          if(response.status == 200){
            let token = response.data.token;
            let user = JSON.stringify(response.data.user);
            console.log(token);
            console.log(user+"in here signin");
            setIsAuthenticated(true);
            setLoginError(false);
            deviceStorage.saveKey("id_token", token);
            deviceStorage.saveKey("user_data", user);
            deviceStorage.loadJWT();
          }
        })
        .catch((err) => {
          console.log(err);
          setLoginError(true);
        });
  };
  const handleSignOut = () => {
    // TODO implement real sign out mechanism
    setLoginError(false);
    setRegisError(false);
    setIsAuthenticated(false);
    deviceStorage.deleteJWT();
    deviceStorage.loadJWT();
    deviceStorage.loadUser();
  };
  const handleSignUp = (firstName, lastName, zipcode, email, password) => {
    console.log("firstName:" , firstName);
    console.log("lastName:" , lastName);
    console.log("zipcode:" , zipcode);
    console.log('email:',email);
    console.log('password:',password);
    // TODO implement real sign up mechanism
    let payload = {
      f_name : firstName,
      l_name : lastName,
      zip : zipcode,
      email_id : email,
      pass : password
    };
    
    axios
    .post(url+"/register", payload, {
      headers: {
        "content-type": "application/json",
      },
    })
    .then((response) => {
      if(response.status == 201){
        //let token = response.data.token;
        //console.log(token);
        setIsAuthenticated(true);
        setRegisError(false);
      }
      else{
        setRegisError(true);
      }
    })
    .catch((err) => {
      console.log(err);
      setRegisError(true);
    });
  };
  return (
      <NavigationContainer>
      <RootStack.Navigator>
        { isAuthenticated ? (
          <React.Fragment>
          <RootStack.Screen name="Home" component={HomeScreen} options={({ navigation, route }) => ({
            headerTitleAlign: 'center',
            headerStyle: {
              backgroundColor: '#003f5c',
            },
            headerTintColor: "#fff",
            headerRight: () => (
              
              <Icon
                name="logout"
                type="material"
                size={40}
                color="#fff"
                onPress={handleSignOut}
                
              />
            ),
            headerLeft: () => (
              <Icon
                name="settings"
                type="material"
                size={40}
                color="#fff"
                style={{ marginLeft: 25}}
                onPress = {() => navigation.navigate('Profile')}
           
              />
            
            ), 
          })}/>
          <RootStack.Screen name="Second" component={SecondScreen} options={{
            headerRight: () => (
              <Icon
                name="logout"
                type="material"
                size={40}
                color="#fff"
                onPress={handleSignOut}
                
              />
            ),
          }}/>

          <RootStack.Screen name="Profile" component={ProfileScreen} options={{
            headerleft: () => (
              <Icon
                name="back"
                type="material"
                size={40}
                color="#fff"  
              />
              
            ),
          }}/>

          </React.Fragment>
        ):
          (<><RootStack.Screen
            name="Landing"
            component={LandingScreen}
          />
          
          <RootStack.Screen name="SignIn">
            {(props) => (
              <SignInScreen {...props} onSignIn={handleSignIn} err={loginError}/>
            )}
          </RootStack.Screen>
          
          <RootStack.Screen name="SignUp">
              {(props) => (
                <SignUpScreen {...props} onSignUp={handleSignUp} err={regisError} />
              )}
            </RootStack.Screen>
          </>)

        }
      </RootStack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#003f5c',
    alignItems: 'center',
    justifyContent: "center",
  },
});

export default App;
